import json
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from _TEMPLATES import apply_mc_template, apply_lgp_grid_template, apply_oeqa_template, apply_gplanet_template

def _auto_find_existing_file(candidates):
    for c in candidates:
        try:
            if c and os.path.exists(c):
                return c
        except Exception:
            pass
    return ""


def _auto_find_local_crux_jsonl() -> str:
    """
    Find a local ZeroEval-format CRUX jsonl (generated from cruxeval) without requiring env vars.
    """
    # 1) Explicit override
    env_path = os.environ.get("ZEROEVAL_CRUX_JSONL", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    # 2) Respect common datasets root env vars
    ds_root = os.environ.get("LOCAL_DATASETS_ROOT", "").strip() or os.environ.get("DATASETS_ROOT", "").strip()
    if ds_root:
        cand = os.path.join(ds_root, "zero-eval", "crux", "test.jsonl")
        if os.path.exists(cand):
            return cand

    # 3) Common relative locations from current working directory
    cwd = os.getcwd()
    rels = [
        os.path.join(cwd, "datasets", "zero-eval", "crux", "test.jsonl"),
        os.path.join(cwd, "..", "datasets", "zero-eval", "crux", "test.jsonl"),
    ]
    p = _auto_find_existing_file(rels)
    if p:
        return os.path.abspath(p)

    # 4) Walk parents from this file location to find `<root>/datasets/zero-eval/crux/test.jsonl`
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        cand = parent / "datasets" / "zero-eval" / "crux" / "test.jsonl"
        if cand.exists():
            return str(cand)

    return ""


def _auto_find_cruxeval_test_jsonl() -> str:
    """
    Find local CRUXEval source jsonl: `<datasets_root>/cruxeval/test.jsonl`.
    """
    ds_root = os.environ.get("LOCAL_DATASETS_ROOT", "").strip() or os.environ.get("DATASETS_ROOT", "").strip()
    cands = []
    if ds_root:
        cands.append(os.path.join(ds_root, "cruxeval", "test.jsonl"))

    cwd = os.getcwd()
    cands += [
        os.path.join(cwd, "datasets", "cruxeval", "test.jsonl"),
        os.path.join(cwd, "..", "datasets", "cruxeval", "test.jsonl"),
    ]

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        cands.append(str(parent / "datasets" / "cruxeval" / "test.jsonl"))

    p = _auto_find_existing_file(cands)
    return os.path.abspath(p) if p else ""


def _make_direct_output_prompt(code: str, input_val: str) -> str:
    return (
        "You are given a Python function and an assertion containing an input to the function. "
        "Complete the assertion with a literal (no unsimplified expressions, no function calls) "
        "containing the output when executing the provided code on the given input, even if the function "
        "is incorrect or incomplete.\n\n"
        "[PYTHON]\n"
        f"{code}\n"
        f"assert f({input_val}) == ??\n"
        "[/PYTHON]\n"
    ).strip()


def _load_crux_from_local_cruxeval_jsonl(cruxeval_jsonl: str):
    """
    Build an in-memory ZeroEval-style dataset from the local CRUXEval jsonl.
    This avoids any Hub access and avoids requiring a pre-generated zero-eval jsonl.
    """
    rows = []
    with open(cruxeval_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                {
                    "id": f"crux-test-#{i}",
                    "question": _make_direct_output_prompt(item.get("code", ""), item.get("input", "")),
                    "answer": item.get("output", ""),
                    "source": "cruxeval-org/cruxeval",
                    "config": "crux",
                    "task_type": "qa",
                }
            )
    return Dataset.from_list(rows)


def mapping_task_names(data_name):
    """
    Mapping the task names to the dataset and id name.
    """
    id_name = "id"
    if data_name == "mmlu-redux":
        dataset = load_dataset("yuchenlin/zero-eval", "mmlu-redux", split="test")
    elif data_name == "gsm":
        dataset = load_dataset("yuchenlin/zero-eval", "gsm", split="test")
    elif data_name == "zebra-grid":
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
    elif data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")  
    elif data_name == "numersense-v2":
        dataset = load_dataset("yuchenlin/zero-eval", "numersense-v2", split="test")
    elif data_name == "crux":
        # Prefer local files to avoid `flydust/zero-eval` Hub dependency (often unreachable in offline mode).
        local_zeroeval_jsonl = _auto_find_local_crux_jsonl()
        if local_zeroeval_jsonl:
            dataset = load_dataset("json", data_files={"test": local_zeroeval_jsonl}, split="test")
        else:
            cruxeval_jsonl = _auto_find_cruxeval_test_jsonl()
            if cruxeval_jsonl:
                dataset = _load_crux_from_local_cruxeval_jsonl(cruxeval_jsonl)
            else:
                dataset = load_dataset("flydust/zero-eval", "crux", split="test")
    elif data_name == "math-l5":
        dataset = load_dataset("AI-MO/aimo-validation-math-level-5", split="train")
    elif data_name == "wildbench_v2-hard":
        dataset = load_dataset("allenai/WildBench", "v2-hard", split="test")
        id_name = "session_id"
    elif data_name == "gplanet":
        dataset = load_dataset("WildEval/G-PlanET", split="test")
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return dataset, id_name

def prompt_generation(data_name, data_item, args):
    """
    Generate prompt for different tasks.
    """
    if data_name in ["mmlu-redux"]:  # and other multiple-choice QA dataset 
        prompt = apply_mc_template(data_item) 
    elif data_name in ["alpaca_eval"]:
        prompt = data_item["instruction"]
    elif data_name in ["wildbench_v2-hard"]:
        prompt = data_item["conversation_input"][0]["content"]
    elif data_name in ["zebra-grid"]:
        prompt = apply_lgp_grid_template(data_item) 
    elif data_name in ["gsm", "math-l5"]:
        question_key = "question"
        if data_name == "math-l5":
            question_key = "problem"
        prompt = apply_oeqa_template(data_item, question_key = question_key)
    elif data_name in ["crux"]:
        prompt = apply_oeqa_template(data_item, cot=True) # cot?
    elif data_name in ["numersense-v2"]:
        if "no_cot" in args.run_name:
            prompt = apply_oeqa_template(data_item, cot=False)
        prompt = apply_oeqa_template(data_item)
    elif data_name in ['gplanet']:
        prompt = apply_gplanet_template(data_item)
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return prompt

def result_format(output_item, args):
    """
    Modify the output format for different tasks if needed.
    """
    if args.data_name in ["alpaca_eval"]:
        output_item["output"] = output_item["output"][0] # use str instead of list 
    elif args.data_name in ["zebra-grid"]:
        if "solution" in output_item:
            del output_item["solution"]
    elif args.data_name in ["wildbench_v2-hard"]:
        for key in ["conversation_input", "references", "length", "checklist", "avg_score", "var_score"]:
            if key in output_item:
                del output_item[key]

    else:
        pass 
    return output_item