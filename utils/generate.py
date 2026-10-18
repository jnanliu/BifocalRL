# coding=utf-8
"""
Generation module for Co-Intuitor using vLLM.

Supports two backends:
1. **Offline API** (``generate_sequences``): HACRL-style — returns token-level data
   (response_ids, old_log_probs, attention_mask) directly. No HTTP server needed.
2. **HTTP API** (``generate_responses``): Legacy — returns text completions via
   vLLM OpenAI-compatible server. Kept for evaluation scripts.
"""

import os
import time
import subprocess
import requests
import logging
import signal
import socket
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union, Any

import psutil

logger = logging.getLogger(__name__)

import re

def _ensure_trust_remote_code_files(checkpoint_dir: str) -> bool:
    """
    Best-effort fix for checkpoints saved from `trust_remote_code=True` models.

    vLLM (and transformers) expect custom python files referenced by `config.json:auto_map`
    (e.g. `modeling_phi3.py`) to be present inside the model folder when loading from a local path.

    Returns True if we copied any missing files into `checkpoint_dir`.
    """
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return False

    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(cfg_path):
        return False

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False

    auto_map = cfg.get("auto_map", None)
    if not isinstance(auto_map, dict) or not auto_map:
        return False

    needed_rel_paths = set()
    for _k, v in auto_map.items():
        values = v if isinstance(v, (list, tuple)) else [v]
        for item in values:
            if not isinstance(item, str):
                continue
            module_path = item.rsplit(".", 1)[0] if "." in item else item
            needed_rel_paths.add(module_path.replace(".", "/") + ".py")

    missing = [p for p in sorted(needed_rel_paths) if not os.path.exists(os.path.join(checkpoint_dir, p))]
    if not missing:
        return False

    source = cfg.get("_name_or_path") or cfg.get("name_or_path")
    if not source:
        return False

    src_root = None
    if isinstance(source, str) and os.path.isdir(source):
        src_root = source
    else:
        try:
            from huggingface_hub import snapshot_download

            src_root = snapshot_download(repo_id=str(source), local_files_only=True)
        except Exception as e:
            logger.warning(
                f"Model folder '{checkpoint_dir}' is missing trust_remote_code files {missing}, "
                f"but could not resolve source repo '{source}' from local cache. Error: {e}"
            )
            return False

    copied_any = False
    for rel_path in missing:
        dst_path = os.path.join(checkpoint_dir, rel_path)

        # Ensure intermediate packages are importable if the module lives in subfolders.
        rel_dir = os.path.dirname(rel_path)
        if rel_dir:
            parts = rel_dir.split("/")
            for i in range(1, len(parts) + 1):
                pkg_rel = "/".join(parts[:i])
                init_rel = os.path.join(pkg_rel, "__init__.py")
                init_dst = os.path.join(checkpoint_dir, init_rel)
                if os.path.exists(init_dst):
                    continue
                init_src = os.path.join(src_root, init_rel)
                os.makedirs(os.path.dirname(init_dst), exist_ok=True)
                try:
                    if os.path.exists(init_src):
                        shutil.copy2(init_src, init_dst)
                    else:
                        with open(init_dst, "w", encoding="utf-8") as f:
                            f.write("")
                except Exception:
                    pass

        src_path = os.path.join(src_root, rel_path)
        if not os.path.exists(src_path):
            alt_src = os.path.join(src_root, os.path.basename(rel_path))
            if os.path.exists(alt_src):
                src_path = alt_src
            else:
                logger.warning(f"Missing remote-code file in source repo: {rel_path} (looked in {src_root})")
                continue

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.copy2(src_path, dst_path)
            copied_any = True
        except Exception as e:
            logger.warning(f"Failed to copy remote-code file {src_path} -> {dst_path}: {e}")

    if copied_any:
        logger.info(f"Patched model folder with trust_remote_code files: {checkpoint_dir}")
    return copied_any

def _should_strip_prompt_echo() -> bool:
    # In practice, some servers/configs may echo the prompt in `choice["text"]`.
    # This breaks downstream logps/scoring by making "completion" extremely long.
    # Default: strip prompt echo when it is a strict prefix. Disable by setting CO_VLLM_STRIP_PROMPT_ECHO=0.
    v = os.environ.get("CO_VLLM_STRIP_PROMPT_ECHO", "1")
    return v not in ("0", "false", "False", "")


def strip_prompt_echo(prompt: str, text: str) -> str:
    """Strip a prompt prefix from a returned text, if it is an exact prefix.

    We only strip on strict prefix match to avoid altering genuine generations.
    """
    try:
        if prompt and text.startswith(prompt):
            return text[len(prompt) :]
    except Exception:
        pass
    return text


_LEADING_CHAT_JUNK_RE = re.compile(
    r"^\s*(?:"
    r"<\|[^>]+\|>"  # ChatML / various HF special tokens
    r"|\[/?INST\]"  # Llama-style
    r"|</s>|<s>"  # sentencepiece-style BOS/EOS markers
    r"|\[/?python\]"  # tool/formatted blocks
    r"|assistant:|Assistant:"  # plain text role markers
    r"|\*\*Answer\*\*:"  # some prompts include this label
    r")\s*",
    re.MULTILINE,
)


def strip_chat_template_artifacts(text: str, *, prompt_rendered: Optional[str] = None) -> str:
    """
    Best-effort cleanup for completions returned by vLLM/OpenAI-style servers.

    Some stacks echo the prompt or include chat-template role markers in the completion.
    Downstream scoring/logps/training expects the completion to be *assistant content only*.
    """
    if text is None:
        return ""
    s = str(text)
    if prompt_rendered and s.startswith(prompt_rendered):
        s = s[len(prompt_rendered) :]
    s = s.strip()
    # Remove a few layers of leading special tokens/role markers.
    for _ in range(8):
        new = _LEADING_CHAT_JUNK_RE.sub("", s, count=1).lstrip()
        if new == s:
            break
        s = new
    return s.strip()


def strip_chat_template_artifacts_from_flattened_responses(
    unique_prompts: List[str],
    flattened_responses: List[str],
    *,
    num_generations: int,
) -> Tuple[List[str], bool]:
    """
    Strip prompt echo + leading chat-template markers from a flattened response list.
    Returns (cleaned_responses, changed).
    """
    cleaned: List[str] = []
    changed = False
    if num_generations <= 0:
        return list(flattened_responses), False
    for i, prompt in enumerate(unique_prompts):
        start = i * num_generations
        end = start + num_generations
        chunk = flattened_responses[start:end]
        for t in chunk:
            t2 = strip_chat_template_artifacts(t, prompt_rendered=str(prompt))
            if t2 != t:
                changed = True
            cleaned.append(t2)
    if len(cleaned) != len(flattened_responses):
        return list(flattened_responses), False
    return cleaned, changed


def strip_prompt_echo_from_flattened_responses(
    unique_prompts: List[str],
    flattened_responses: List[str],
    *,
    num_generations: int,
) -> Tuple[List[str], bool]:
    """Strip prompt echo from a flattened response list (prompt-major then generation-major).

    Returns (cleaned_responses, changed).
    """
    cleaned: List[str] = []
    changed = False
    if num_generations <= 0:
        return list(flattened_responses), False
    for i, prompt in enumerate(unique_prompts):
        start = i * num_generations
        end = start + num_generations
        chunk = flattened_responses[start:end]
        for t in chunk:
            t2 = strip_prompt_echo(prompt, str(t))
            if t2 != t:
                changed = True
            cleaned.append(t2)
    # If lengths mismatch, fall back to original without crashing.
    if len(cleaned) != len(flattened_responses):
        return list(flattened_responses), False
    return cleaned, changed


class OpenAIClient:
    """OpenAI协议适配器，用于调用vLLM服务器"""
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}/v1"
    
    def generate(self, prompts: List[str], **kwargs) -> List[dict]:
        """调用 /v1/completions 生成文本"""
        responses = []
        url = f"{self.base_url}/completions"
        
        params = {
            "model": kwargs.get("model", "default"),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "n": kwargs.get("n", 1),
            # Avoid empty generations when the model immediately emits EOS/stop.
            # vLLM OpenAI server supports `min_tokens` (ignored if unsupported).
            "min_tokens": kwargs.get("min_tokens", 1),
        }

        seed = kwargs.get("seed", None)
        if seed is not None:
            params["seed"] = int(seed)

        # vLLM OpenAI server supports repetition_penalty (OpenAI-compatible servers may ignore it).
        repetition_penalty = kwargs.get("repetition_penalty", None)
        if repetition_penalty is not None:
            try:
                rp = float(repetition_penalty)
                # Only send when meaningfully different from default.
                if rp != 1.0:
                    params["repetition_penalty"] = rp
            except Exception:
                # Ignore invalid values to keep generation robust.
                pass

        # Optional stop sequences (open-r1 does not use this by default).
        stop = kwargs.get("stop", None)
        if stop is not None:
            params["stop"] = stop
        
        params["prompt"] = prompts
        
        try:
            resp = requests.post(url, json=params, timeout=kwargs.get("timeout", 600))
            if resp.status_code != 200:
                raise RuntimeError(f"Request failed: {resp.status_code}, {resp.text}")
            
            data = resp.json()
            if isinstance(data, dict) and data.get("error") is not None:
                raise RuntimeError(f"vLLM error: {data['error']}")
            return data["choices"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Generation request failed: {e}")


def check_server_health(host: str, port: int, timeout: int = 5) -> bool:
    """检查服务器健康状态"""
    url = f"http://{host}:{port}/health"
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def is_sleeping(host: str, port: int) -> Optional[bool]:
    """检查服务器是否在休眠"""
    url = f"http://{host}:{port}/is_sleeping"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return response.json().get("is_sleeping", False)
        return None
    except requests.exceptions.RequestException:
        return None


def wake_server(host: str, port: int, timeout: int = 120) -> bool:
    """唤醒服务器"""
    url = f"http://{host}:{port}/wake_up"
    try:
        requests.post(url, timeout=10)
        logger.info("Waking up server...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if check_server_health(host, port) and is_sleeping(host, port) is False:
                logger.info("Server is awake!")
                return True
            time.sleep(2)
        logger.error("Timeout waiting for server to wake up!")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Wake-up request error: {e}")
        return False


def sleep_server_api(host: str, port: int, level: int = 1) -> bool:
    """让服务器休眠"""
    url = f"http://{host}:{port}/sleep"
    try:
        response = requests.post(url, params={"level": level}, timeout=5)
        if response.status_code == 200:
            logger.info("Server sleep command sent successfully.")
            return True
        return False
    except requests.exceptions.RequestException:
        return False


def start_vllm_server(
    model_path: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    gpu_memory_utilization: float = 0.9,
    dtype: str = "bfloat16",
    cuda_device: int = 0,
    log_file: str = "vllm_server.log"
) -> subprocess.Popen:
    """启动vLLM服务器"""
    # If this is a local checkpoint missing remote-code python files (e.g. modeling_phi3.py),
    # patch it from HF cache before vLLM loads it.
    try:
        _ensure_trust_remote_code_files(model_path)
    except Exception:
        # Best-effort; don't block server start.
        pass

    # Optional performance knobs (override via env):
    # - VLLM_MAX_MODEL_LEN: cap context to reduce KV cache blow-ups (esp. if model advertises huge max_seq_len)
    # - VLLM_MAX_NUM_SEQS: cap concurrency to keep KV cache from hitting 100%
    # - VLLM_MAX_NUM_BATCHED_TOKENS: chunked prefill budget
    # - VLLM_ENABLE_CHUNKED_PREFILL: "true"|"false"
    max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN", "").strip()
    max_num_seqs = os.environ.get("VLLM_MAX_NUM_SEQS", "").strip()
    max_num_batched_tokens = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "").strip()
    enable_chunked_prefill = os.environ.get("VLLM_ENABLE_CHUNKED_PREFILL", "true").strip().lower()

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", host,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--enable-sleep-mode",
        "--trust-remote-code",
    ]

    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if max_num_seqs:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    if max_num_batched_tokens:
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

    if enable_chunked_prefill in ("0", "false", "no", "off"):
        cmd.append("--disable-chunked-prefill")
    else:
        cmd.append("--enable-chunked-prefill")

    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env["VLLM_SERVER_DEV_MODE"] = "1"

    logger.info(f"Starting server: {' '.join(cmd)}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            # Put vLLM in its own process group so we can kill all children.
            start_new_session=True,
        )
    return process


def _is_port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """Returns True if something is listening on host:port (tries all resolved addrs)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        infos = []
    for family, socktype, proto, _canon, sockaddr in infos:
        try:
            s = socket.socket(family, socktype, proto)
            try:
                s.settimeout(timeout_s)
                s.connect(sockaddr)
                return True
            finally:
                s.close()
        except OSError:
            continue
    return False


def is_port_in_use(port: int, hosts: Optional[List[str]] = None, timeout_s: float = 0.5) -> bool:
    """
    Best-effort "is something listening on this port?" check.

    We probe multiple common loopback hosts to avoid IPv4/IPv6 resolution mismatches
    (e.g. a service bound on 127.0.0.1 while `localhost` resolves to ::1).
    """
    hs = hosts or ["127.0.0.1", "localhost", "::1"]
    for h in hs:
        if _is_port_open(h, int(port), timeout_s=timeout_s):
            return True
    return False


def pick_free_port(preferred_port: int, *, max_tries: int = 50) -> int:
    """
    Pick an available TCP port, starting from `preferred_port` and scanning upwards.
    """
    p0 = int(preferred_port)
    for p in range(p0, p0 + int(max_tries)):
        if not is_port_in_use(p):
            return p
    # Fall back to an ephemeral port.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _wait_for_port_close(host: str, port: int, timeout_s: float = 30.0) -> bool:
    """Wait until host:port stops accepting TCP connections."""
    start = time.time()
    while time.time() - start < timeout_s:
        if not _is_port_open(host, port, timeout_s=0.5):
            return True
        time.sleep(0.5)
    return False


def stop_vllm_server_on_port(host: str, port: int, timeout_s: float = 60.0) -> bool:
    """Force-stop a vLLM server by killing the process that listens on host:port.

    This is stronger than sleep mode: it fully frees GPU memory by terminating the server.
    It is safe to call even if nothing is listening (returns True).
    """
    # If nothing is listening, we're already "stopped".
    if not _is_port_open(host, port, timeout_s=0.5):
        return True

    # Map TCP port -> pid using psutil (cross-platform-ish, no shell dependency).
    pid: Optional[int] = None
    try:
        for c in psutil.net_connections(kind="tcp"):
            try:
                laddr = getattr(c, "laddr", None)
                if not laddr:
                    continue
                if int(getattr(laddr, "port", -1)) != int(port):
                    continue
                if c.status != psutil.CONN_LISTEN:
                    continue
                if c.pid is None:
                    continue
                pid = int(c.pid)
                break
            except Exception:
                continue
    except Exception:
        pid = None

    if pid is None:
        # Best-effort fallback: if port is open but we can't resolve PID, don't crash.
        logger.warning(f"Port {host}:{port} appears open but PID could not be resolved.")
        return False

    try:
        proc = psutil.Process(pid)
    except Exception:
        # PID vanished; re-check port.
        return not _is_port_open(host, port, timeout_s=0.5)

    try:
        # Terminate children first (vLLM may spawn workers).
        children = []
        try:
            children = proc.children(recursive=True)
        except Exception:
            children = []
        for ch in children:
            try:
                ch.terminate()
            except Exception:
                pass
        try:
            proc.terminate()
        except Exception:
            pass

        # Wait a bit, then escalate.
        gone, alive = psutil.wait_procs([proc] + children, timeout=15)
        if alive:
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass
            psutil.wait_procs(alive, timeout=15)
    except Exception as e:
        logger.warning(f"Failed to terminate process on port {host}:{port} (pid={pid}): {e}")

    return _wait_for_port_close(host, port, timeout_s=timeout_s)


def wait_for_vllm_server(url: str, timeout: int = 600) -> bool:
    """等待vLLM服务器启动"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def stop_vllm_server(process: subprocess.Popen):
    """停止vLLM服务器"""
    if process:
        try:
            logger.info("Terminating vLLM server process...")
            # Prefer killing the whole process group (vLLM may spawn children).
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    process.kill()
        except Exception as e:
            logger.error(f"Error stopping process: {e}")


def generate_responses(
    model_path: str,
    prompts: List[str],
    num_generations: int = 4,
    batch_size: int = 1,
    port: int = 8000,
    host: str = "localhost",
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    repetition_penalty: float = 1.0,
    stop: Optional[Union[str, List[str]]] = None,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "bfloat16",
    cuda_device: Optional[int] = None,
    auto_start_server: bool = True,
    auto_stop_server: bool = True,
    return_finish_reasons: bool = False,
    seed: Optional[int] = None,
) -> Any:
    """
    使用vLLM生成响应（不需要accelerator）。
    
    Args:
        model_path: 模型路径
        prompts: 提示文本列表
        num_generations: 每个prompt生成的数量
        batch_size: 批次大小
        port: 服务器端口
        host: 服务器主机
        temperature: 生成温度
        top_p: Top-p采样
        max_tokens: 最大token数
        gpu_memory_utilization: GPU内存利用率
        dtype: 数据类型
        cuda_device: CUDA设备ID
        auto_start_server: 是否自动启动服务器
        auto_stop_server: 是否自动停止服务器
    
    Returns:
        If return_finish_reasons is False:
          - responses: 扁平化 completion 文本列表
        If return_finish_reasons is True:
          - (responses, finish_reasons): finish_reasons 为扁平化列表（与 responses 对齐），元素为 vLLM 的 choice.finish_reason
    """
    server_process = None
    log_file = f"vllm_server_{port}.log"
    connect_host = "localhost" if host == "0.0.0.0" else host

    try:
        if auto_start_server:
            logger.info("Auto-starting vLLM server...")
            # If something is already listening (possibly on IPv4 while localhost resolves to IPv6),
            # don't hard-fail: pick a free port to avoid "address already in use".
            if is_port_in_use(port):
                new_port = pick_free_port(port + 1)
                logger.warning(
                    f"Port {port} is already in use. Switching to free port {new_port}."
                )
                port = int(new_port)
                log_file = f"vllm_server_{port}.log"
            server_process = start_vllm_server(
                model_path=model_path,
                port=port,
                host=host,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                cuda_device=cuda_device if cuda_device is not None else 0,
                log_file=log_file
            )
            
            if not wait_for_vllm_server(f"http://{connect_host}:{port}/health", timeout=600):
                raise RuntimeError("Failed to start vLLM server")
        else:
            logger.info(f"Connecting to existing server at {connect_host}:{port}...")
            if not check_server_health(connect_host, port):
                raise RuntimeError(f"Server at {connect_host}:{port} is not reachable")

            sleeping = is_sleeping(connect_host, port)
            if sleeping is True:
                logger.info("Waking up sleeping server...")
                if not wake_server(connect_host, port):
                    raise RuntimeError("Failed to wake up server")

        logger.info(f"Generating {len(prompts)} prompts...")
        client = OpenAIClient(connect_host, port)
        
        all_responses: List[str] = []
        all_finish_reasons: List[Optional[str]] = []
        from tqdm import tqdm
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        with tqdm(total=len(prompts), desc="Generating responses", unit="prompt") as pbar:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                
                choices = client.generate(
                    prompts=batch_prompts,
                    model=model_path,
                    n=num_generations,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    stop=stop,
                    min_tokens=1,
                    timeout=900,
                    seed=seed,
                )
                
                expected = len(batch_prompts) * num_generations
                if len(choices) != expected:
                    raise RuntimeError(
                        f"vLLM returned unexpected number of choices: got {len(choices)}, expected {expected} "
                        f"(batch_prompts={len(batch_prompts)}, n={num_generations})"
                    )

                # Extract only the completion part
                # vLLM's /v1/completions API returns choices in order:
                # For batch_size=2, n=4: choices[0-3] are for prompt[0], choices[4-7] are for prompt[1]
                # Each choice["text"] should contain only the completion (not the prompt)
                # But we need to handle cases where it might include the prompt
                # Organize responses per prompt to support targeted retries for empty generations.
                per_prompt: List[List[str]] = [[] for _ in range(len(batch_prompts))]
                per_prompt_finish: List[List[Optional[str]]] = [[] for _ in range(len(batch_prompts))]
                empty_slots: List[tuple[int, int]] = []  # (prompt_idx, gen_idx)

                for j, choice in enumerate(choices):
                    idx = choice.get("index", j)
                    prompt_idx = int(idx) // num_generations
                    gen_idx = int(idx) % num_generations
                    if prompt_idx >= len(batch_prompts):
                        # Fallback to sequential mapping
                        prompt_idx = j // num_generations
                        gen_idx = j % num_generations

                    response_text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason", None)

                    # If response contains the prompt, remove it (echo).
                    original_prompt = batch_prompts[prompt_idx]
                    if _should_strip_prompt_echo():
                        response_text = strip_prompt_echo(original_prompt, str(response_text))

                    # IMPORTANT: do NOT strip the completion text.
                    #
                    # We later reconstruct `completion_ids` by tokenizing the cached text (open-r1 style).
                    # Stripping can change tokenization (especially leading whitespace/newlines), making cache
                    # not perfectly reversible.
                    response_text = str(response_text)

                    # Ensure list has enough slots
                    while len(per_prompt[prompt_idx]) < num_generations:
                        per_prompt[prompt_idx].append("")
                    while len(per_prompt_finish[prompt_idx]) < num_generations:
                        per_prompt_finish[prompt_idx].append(None)
                    per_prompt[prompt_idx][gen_idx] = response_text
                    per_prompt_finish[prompt_idx][gen_idx] = finish_reason
                    if response_text.strip() == "":
                        empty_slots.append((prompt_idx, gen_idx))

                # Retry empty generations a few times, batched by n value.
                max_empty_retries = 3
                for attempt in range(max_empty_retries):
                    if not empty_slots:
                        break
                    to_retry: Dict[int, int] = {}
                    for p_idx, _g_idx in empty_slots:
                        to_retry[p_idx] = to_retry.get(p_idx, 0) + 1

                    empty_slots = []

                    # Group prompts by missing_n so we can batch them in one request per group.
                    n_groups: Dict[int, List[int]] = defaultdict(list)
                    for p_idx, missing_n in to_retry.items():
                        n_groups[missing_n].append(p_idx)

                    for missing_n, p_indices in n_groups.items():
                        group_prompts = [batch_prompts[p_idx] for p_idx in p_indices]
                        logger.warning(
                            f"Empty generations detected ({len(p_indices)} prompts, missing_n={missing_n}). "
                            f"Retrying as batch (attempt {attempt+1}/{max_empty_retries})..."
                        )
                        retry_choices = client.generate(
                            prompts=group_prompts,
                            model=model_path,
                            n=missing_n,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            repetition_penalty=repetition_penalty,
                            stop=stop,
                            min_tokens=1,
                            timeout=900,
                            seed=seed,
                        )
                        # Distribute results back: choices are ordered
                        # [p0_g0..p0_g{n-1}, p1_g0..p1_g{n-1}, ...]
                        for group_i, p_idx in enumerate(p_indices):
                            prompt_text = batch_prompts[p_idx]
                            start_c = group_i * missing_n
                            end_c = start_c + missing_n
                            for c in retry_choices[start_c:end_c]:
                                t = c.get("text", "")
                                fr = c.get("finish_reason", None)
                                if _should_strip_prompt_echo():
                                    t = strip_prompt_echo(prompt_text, str(t))
                                t = str(t)
                                for k in range(num_generations):
                                    if per_prompt[p_idx][k] == "":
                                        per_prompt[p_idx][k] = t
                                        per_prompt_finish[p_idx][k] = fr
                                        break
                            for k in range(num_generations):
                                if per_prompt[p_idx][k] == "":
                                    empty_slots.append((p_idx, k))

                # Append to flattened list in the same order as prompts then gens.
                for p_idx in range(len(batch_prompts)):
                    for g_idx in range(num_generations):
                        response_text = per_prompt[p_idx][g_idx]
                        if response_text.strip() == "":
                            logger.error(f"Still empty response after retries (prompt_idx={p_idx}, gen_idx={g_idx}). Keeping empty string.")
                        all_responses.append(response_text)
                        all_finish_reasons.append(per_prompt_finish[p_idx][g_idx] if p_idx < len(per_prompt_finish) else None)
                
                # Update progress bar (each batch processes batch_size prompts)
                pbar.update(len(batch_prompts))
        
        if return_finish_reasons:
            return all_responses, all_finish_reasons
        return all_responses
        
    finally:
        if auto_stop_server:
            logger.info("Stopping vLLM server (Auto-Stop)...")
            stop_vllm_server(server_process)
            # Ensure the port is actually released before returning.
            if server_process is not None:
                if not _wait_for_port_close(connect_host, port, timeout_s=60.0):
                    logger.warning(f"Port {connect_host}:{port} still appears open after stopping vLLM server.")
        elif not auto_start_server:
            logger.info("Putting server to sleep (Smart Resource Management)...")
            sleep_server_api(connect_host, port, level=1)


# ---------------------------------------------------------------------------
# HACRL-style offline generation via vLLM Python API
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutput:
    """Structured token-level output from vLLM generation (HACRL-style).

    All lists are flattened with prompt-major ordering:
    ``[prompt_0_gen_0, prompt_0_gen_1, ..., prompt_1_gen_0, ...]``.
    ``response_ids`` and ``old_log_probs`` are variable-length per sample;
    use :meth:`pad_old_logps` to get a uniform-width matrix for caching.
    """
    prompt_ids: List[List[int]] = field(default_factory=list)
    response_ids: List[List[int]] = field(default_factory=list)
    response_texts: List[str] = field(default_factory=list)
    old_log_probs: List[List[float]] = field(default_factory=list)
    finish_reasons: List[str] = field(default_factory=list)

    def pad_old_logps(self, pad_value: float = 0.0) -> Tuple[List[List[float]], int]:
        """Pad ``old_log_probs`` to uniform *logits_to_keep* width.

        Returns:
            ``(padded_rows, logits_to_keep)`` where each row has length
            *logits_to_keep*.
        """
        if not self.old_log_probs:
            return [], 0
        ltk = max((len(lp) for lp in self.old_log_probs), default=1)
        ltk = max(ltk, 1)
        padded = [lp + [pad_value] * (ltk - len(lp)) for lp in self.old_log_probs]
        return padded, ltk


def create_offline_llm(
    model_path: str,
    *,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    cuda_device: Optional[int] = None,
    seed: Optional[int] = None,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    enable_chunked_prefill: Optional[bool] = None,
    enforce_eager: Optional[bool] = None,
) -> Any:
    """Create and return a reusable vLLM offline engine.

    When *enforce_eager* is ``None`` (default), it defaults to ``False``
    so that vLLM can use CUDA Graphs for better performance.  Generation
    should be run outside ``accelerate`` (standalone process) to avoid
    vLLM v1 sub-process NCCL conflicts.
    """
    from vllm import LLM

    try:
        from utils.model_utils import (
            patch_ministral3_config,
            patch_vllm_qwen35_text_registry,
        )
        patch_ministral3_config()
        patch_vllm_qwen35_text_registry()
    except Exception:
        pass

    try:
        _ensure_trust_remote_code_files(model_path)
    except Exception:
        pass

    if enforce_eager is None:
        enforce_eager = False

    saved_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    dist_env_keys = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
        "GROUP_RANK",
    )
    saved_dist_env = {k: os.environ.get(k) for k in dist_env_keys}
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    # generate_only runs without torchrun/accelerate. If these vars leak from a
    # previous distributed stage, vLLM's torch.distributed init can reuse a stale
    # rendezvous endpoint and fail with EADDRINUSE.
    for k in dist_env_keys:
        os.environ.pop(k, None)

    try:
        llm_kwargs: Dict[str, Any] = dict(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
        )
        if seed is not None:
            llm_kwargs["seed"] = int(seed)
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if enable_chunked_prefill is not None:
            llm_kwargs["enable_chunked_prefill"] = bool(enable_chunked_prefill)

        logger.info(
            "Loading reusable vLLM offline model: %s (tp=%d, dtype=%s, gpu_mem=%.2f, max_model_len=%s, max_num_seqs=%s, max_num_batched_tokens=%s, chunked_prefill=%s, enforce_eager=%s)",
            model_path,
            tensor_parallel_size,
            dtype,
            gpu_memory_utilization,
            max_model_len,
            max_num_seqs,
            max_num_batched_tokens,
            enable_chunked_prefill,
            enforce_eager,
        )
        return LLM(**llm_kwargs)
    finally:
        if saved_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda
        elif cuda_device is not None and "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        for k, v in saved_dist_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def shutdown_offline_llm(llm: Optional[Any]) -> None:
    """Best-effort shutdown for vLLM offline engine and workers."""
    if llm is None:
        return

    import gc
    import torch

    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_sequences(
    model_path: str,
    prompts: List[str],
    *,
    num_generations: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    repetition_penalty: float = 1.0,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    cuda_device: Optional[int] = None,
    seed: Optional[int] = None,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    enable_chunked_prefill: Optional[bool] = None,
    llm: Optional[Any] = None,
    enforce_eager: Optional[bool] = None,
) -> GenerationOutput:
    """Generate sequences using the vLLM **offline** Python API.

    This mirrors the HACRL/verl ``generate_sequences`` semantics: the caller
    receives exact token IDs and per-token log-probabilities computed during
    generation (using the KV cache), eliminating the text round-trip and the
    need for a separate ``compute_logps`` forward pass.

    Args:
        model_path: Local model checkpoint or HuggingFace repo id.
        prompts: Text prompts (already rendered with chat template).
        num_generations: Completions per prompt (``n``).
        temperature: Sampling temperature (0 = greedy decoding in vLLM).
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum response length in tokens.
        repetition_penalty: vLLM repetition penalty (1.0 = disabled).
        gpu_memory_utilization: Fraction of GPU memory for KV cache.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        dtype: Model dtype (``"bfloat16"``, ``"float16"``, ``"auto"``).
        cuda_device: If set, restricts to this CUDA device via env var.
        seed: Random seed for reproducible sampling.
        trust_remote_code: Passed to vLLM ``LLM`` constructor.
        max_model_len: Optional context-length cap (reduces KV cache size).
        max_num_seqs: Optional cap on concurrent in-flight sequences.
        max_num_batched_tokens: Optional cap on total tokens per vLLM batch.
        enable_chunked_prefill: Optional chunked-prefill override for vLLM.

    Returns:
        :class:`GenerationOutput` with token-level data.
    """
    from vllm import SamplingParams

    try:
        _ensure_trust_remote_code_files(model_path)
    except Exception:
        pass

    own_llm = llm is None
    saved_cuda = None
    if own_llm:
        saved_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    try:
        if llm is None:
            llm = create_offline_llm(
                model_path=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                cuda_device=cuda_device,
                seed=seed,
                trust_remote_code=trust_remote_code,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=enable_chunked_prefill,
                enforce_eager=enforce_eager,
            )

        sp_kwargs: Dict[str, Any] = dict(
            n=num_generations,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            logprobs=1,
            min_tokens=1,
        )
        if repetition_penalty != 1.0:
            sp_kwargs["repetition_penalty"] = float(repetition_penalty)
        if seed is not None:
            sp_kwargs["seed"] = int(seed)

        sampling_params = SamplingParams(**sp_kwargs)

        logger.info(
            "Generating %d prompts × %d completions (temp=%.2f, top_p=%.2f, max_tokens=%d) ...",
            len(prompts), num_generations, temperature, top_p, max_tokens,
        )
        outputs = llm.generate(prompts, sampling_params)

        result = GenerationOutput()

        for req_output in outputs:
            prompt_token_ids = list(req_output.prompt_token_ids)
            for comp in req_output.outputs:
                result.prompt_ids.append(prompt_token_ids)
                result.response_ids.append(list(comp.token_ids))
                result.response_texts.append(comp.text)
                result.finish_reasons.append(comp.finish_reason or "stop")

                if comp.logprobs is not None:
                    token_logps: List[float] = []
                    for step_idx, step_dict in enumerate(comp.logprobs):
                        sampled_id = comp.token_ids[step_idx]
                        if step_dict and sampled_id in step_dict:
                            token_logps.append(float(step_dict[sampled_id].logprob))
                        else:
                            token_logps.append(0.0)
                    result.old_log_probs.append(token_logps)
                else:
                    result.old_log_probs.append([0.0] * len(comp.token_ids))

        expected = len(prompts) * num_generations
        if len(result.response_texts) != expected:
            raise RuntimeError(
                f"vLLM returned {len(result.response_texts)} completions, expected {expected}"
            )
        logger.info("Generation complete: %d completions", len(result.response_texts))

        return result

    finally:
        if own_llm:
            shutdown_offline_llm(llm)

            if saved_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda
            elif cuda_device is not None and "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
