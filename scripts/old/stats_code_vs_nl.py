#!/usr/bin/env python3
# coding: utf-8
"""
统计每个模型每次生成中，使用代码 vs 自然语言解决问题的数量和准确率。

支持数据源:
1. ZeroEval parsed JSON (CRUXEval 等) - 推荐，含 raw_output
2. Custom eval runner 输出 (eval/runner 的 results_*.json) - 含 responses[].text
3. LightEval details parquet - 部分版本可能只存 token ids，无法提取文本

注意: LightEval 的 --save-details 若只存储 cont_tokens 而非解码文本，
则无法分类 code/nl，需用 custom eval 或 ZeroEval 获取带 raw_output 的数据。

LightEval 内置指标（两种方式）:
  1) 用本地任务: --custom_tasks eval.lighteval.custom_tasks.offline_local_tasks
     + --map_builtin_math_500_to_custom --map_builtin_gsm8k_to_custom
  2) 用默认任务: --custom_tasks eval.lighteval.custom_tasks.builtin_patch_code_nl
     + --no-map_builtin_math_500_to_custom --no-map_builtin_gsm8k_to_custom
  均需设置 CO_LIGHTEVAL_CODE_NL_METRICS=1，results 中会有
  co_math_code_accuracy, co_math_natural_language_accuracy 等。

用法:
  # 分析 LightEval 结果
  python scripts/stats_code_vs_nl.py --lighteval-root debug/lighteval

  # 分析 ZeroEval 结果
  python scripts/stats_code_vs_nl.py --zeroeval-root debug/zeroeval

  # 分析 custom eval 结果
  python scripts/stats_code_vs_nl.py --eval-root outputs/eval

  # 指定输出
  python scripts/stats_code_vs_nl.py --lighteval-root ... --output stats_code_nl.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CodeVsNLStats:
    """Per-model, per-task stats for code vs natural language."""

    code_count: int = 0
    code_correct: int = 0
    nl_count: int = 0
    nl_correct: int = 0
    unknown_count: int = 0
    unknown_correct: int = 0

    @property
    def code_accuracy(self) -> Optional[float]:
        if self.code_count == 0:
            return None
        return self.code_correct / self.code_count

    @property
    def nl_accuracy(self) -> Optional[float]:
        if self.nl_count == 0:
            return None
        return self.nl_correct / self.nl_count

    @property
    def total(self) -> int:
        return self.code_count + self.nl_count + self.unknown_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": {"count": self.code_count, "correct": self.code_correct, "accuracy": self.code_accuracy},
            "natural_language": {"count": self.nl_count, "correct": self.nl_correct, "accuracy": self.nl_accuracy},
            "unknown": {"count": self.unknown_count, "correct": self.unknown_correct},
            "total": self.total,
        }


def _load_parquet(path: Path):
    """Load parquet file; try datasets first, fallback to pandas."""
    try:
        from datasets import load_dataset

        return load_dataset("parquet", data_files=str(path), split="train")
    except Exception:
        pass
    try:
        import pandas as pd

        return pd.read_parquet(path)
    except Exception:
        # pyarrow direct
        try:
            import pyarrow.parquet as pq

            return pq.read_table(path)
        except Exception:
            return None


def _extract_generation_from_lighteval_row(row: Any) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract (generation_text, metric_value) from a LightEval detail row.
    Returns (None, None) if extraction fails.
    LightEval may store: __model_response__ with generations/cont_tokens/cont.
    """
    text = None
    metric = None

    def get_metric(r):
        m = r.get("__metric__") if hasattr(r, "get") else None
        if m is not None:
            try:
                return float(m)
            except (TypeError, ValueError):
                return 1.0 if m else 0.0
        return None

    def get_text_from_resp(resp):
        if resp is None:
            return None
        if isinstance(resp, str) and len(resp.strip()) > 0:
            return resp
        if isinstance(resp, dict):
            for key in ("generations", "generation", "cont", "text", "completions"):
                v = resp.get(key)
                if v is not None:
                    if isinstance(v, list) and v:
                        v = v[0]
                    if isinstance(v, str) and v.strip():
                        return v
                    if isinstance(v, list) and v and isinstance(v[0], str):
                        return v[0]
        if isinstance(resp, list) and resp:
            first = resp[0]
            if isinstance(first, str) and first.strip():
                return first
            if isinstance(first, dict):
                return get_text_from_resp(first)
        return None

    if hasattr(row, "get"):
        metric = get_metric(row)
        resp = row.get("__model_response__")
        text = get_text_from_resp(resp)

    return (text, metric)


def _extract_gold_from_doc(doc: Any) -> Optional[str]:
    """Extract gold answer from LightEval __doc__."""
    if doc is None:
        return None
    if isinstance(doc, dict):
        spec = doc.get("specific")
        if isinstance(spec, dict) and spec.get("gold") is not None:
            return str(spec["gold"])
        choices = doc.get("choices")
        if isinstance(choices, list) and choices:
            return str(choices[0])
    return None


def _load_lighteval_details(parquet_path: Path) -> List[Tuple[str, float, Optional[str]]]:
    """
    Load LightEval details parquet.
    Returns list of (generation_text, metric_value, gold).
    metric_value: 1.0 = correct, 0.0 = wrong (from __metric__).
    """
    data = _load_parquet(parquet_path)
    if data is None:
        return []

    rows: List[Tuple[str, float, Optional[str]]] = []
    n = len(data)

    for i in range(n):
        row = data[i]
        text, metric = _extract_generation_from_lighteval_row(row)
        if metric is None:
            metric = 0.0
        gold = None
        if hasattr(row, "get"):
            doc = row.get("__doc__")
            gold = _extract_gold_from_doc(doc)

        if text is None:
            # Some LightEval versions store cont_tokens; we'd need tokenizer to decode.
            # For now, we can't classify - skip or mark unknown.
            continue
        rows.append((text, metric, gold))
    return rows


def _load_zeroeval_parsed(json_path: Path) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Load ZeroEval parsed JSON (e.g. parsed_checkpoint-*_model-a.json).
    Returns list of (raw_output, matched, gold).
    """
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows: List[Tuple[str, bool, Optional[str]]] = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                raw = item.get("raw_output") or item.get("output") or ""
                matched = item.get("matched", False)
                gold = item.get("gold")
                rows.append((str(raw), bool(matched), str(gold) if gold is not None else None))
    return rows


def _load_custom_eval_results(json_path: Path) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Load custom eval runner output (eval/runner format).
    Returns list of (text, is_correct, gold) per response (flattened).
    """
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows: List[Tuple[str, bool, Optional[str]]] = []
    problems = obj.get("problems", [])
    for prob in problems:
        gold = prob.get("gold") or prob.get("gold_used")
        for resp in prob.get("responses", []):
            text = resp.get("text", "")
            ok = resp.get("is_correct", False)
            rows.append((str(text), bool(ok), str(gold) if gold is not None else None))
    return rows


def _collect_lighteval_parquets(root: Path) -> List[Tuple[str, str, Path]]:
    """
    Find all LightEval detail parquet files.
    Returns list of (model_id, task_name, path).
    model_id: relative path from root (e.g. co-wmv-sl/model-b/checkpoint-10)
    """
    out: List[Tuple[str, str, Path]] = []
    root = root.resolve()
    for p in root.rglob("details_*.parquet"):
        if not p.is_file():
            continue
        # Path like: .../checkpoint-10/details/.../details_lighteval|gsm8k|0_2026-02-23T....parquet
        name = p.stem
        if name.startswith("details_"):
            task_part = name[8:]  # remove "details_"
            if "_" in task_part:
                parts = task_part.rsplit("_", 1)
                task_name = parts[0] if len(parts) >= 2 else task_part
            else:
                task_name = task_part
            try:
                model_id = str(p.relative_to(root).parent.parent.parent)
            except ValueError:
                model_id = p.parent.parent.name
            out.append((model_id, task_name, p))
    return out


def _collect_zeroeval_parsed(root: Path) -> List[Tuple[str, str, Path]]:
    """
    Find ZeroEval parsed JSON files.
    Returns list of (model_id, task_name, path).
    """
    out: List[Tuple[str, str, Path]] = []
    for p in root.rglob("parsed_*.json"):
        if not p.is_file():
            continue
        # parsed_checkpoint-*_model-a.json or similar
        stem = p.stem
        if stem.startswith("parsed_"):
            rest = stem[7:]
            parts = rest.split("_model-")
            model_id = parts[-1] if len(parts) >= 2 else "unknown"
            task_name = p.parent.name if p.parent.name else "cruxeval"
            out.append((model_id, task_name, p))
    return out


def _collect_custom_eval(root: Path) -> List[Tuple[str, str, Path]]:
    """
    Find custom eval result JSON files.
    Returns list of (model_id, task_name, path).
    """
    out: List[Tuple[str, str, Path]] = []
    for p in root.rglob("results_*.json"):
        if not p.is_file():
            continue
        # results_checkpoint-*_model-a.json
        stem = p.stem
        if stem.startswith("results_"):
            rest = stem[8:]
            parts = rest.split("_model-")
            model_id = parts[-1] if len(parts) >= 2 else "unknown"
            task_name = p.parent.name if p.parent.name else "unknown"
            out.append((model_id, task_name, p))
    return out


def _run_stats(
    lighteval_root: Optional[Path] = None,
    zeroeval_root: Optional[Path] = None,
    eval_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run aggregation and return report dict."""
    from eval.common.code_vs_nl import classify_generation

    report: Dict[str, Any] = {
        "sources": [],
        "by_model": {},
        "summary": {},
    }

    all_stats: Dict[str, Dict[str, CodeVsNLStats]] = {}  # model_id -> task -> stats

    def ensure_model_task(model_id: str, task: str) -> CodeVsNLStats:
        if model_id not in all_stats:
            all_stats[model_id] = {}
        if task not in all_stats[model_id]:
            all_stats[model_id][task] = CodeVsNLStats()
        return all_stats[model_id][task]

    # 1. LightEval details
    if lighteval_root and lighteval_root.exists():
        parquets = _collect_lighteval_parquets(lighteval_root)
        report["sources"].append({"type": "lighteval", "count": len(parquets)})
        for model_id, task_name, path in parquets:
            rows = _load_lighteval_details(path)
            if not rows:
                continue
            st = ensure_model_task(model_id, task_name)
            for text, metric, _gold in rows:
                kind = classify_generation(text)
                is_correct = metric >= 0.5
                if kind == "code":
                    st.code_count += 1
                    if is_correct:
                        st.code_correct += 1
                elif kind == "natural_language":
                    st.nl_count += 1
                    if is_correct:
                        st.nl_correct += 1
                else:
                    st.unknown_count += 1
                    if is_correct:
                        st.unknown_correct += 1

    # 2. ZeroEval parsed
    if zeroeval_root and zeroeval_root.exists():
        parsed_files = _collect_zeroeval_parsed(zeroeval_root)
        report["sources"].append({"type": "zeroeval", "count": len(parsed_files)})
        for model_id, task_name, path in parsed_files:
            rows = _load_zeroeval_parsed(path)
            if not rows:
                continue
            st = ensure_model_task(model_id, task_name)
            for text, matched, _gold in rows:
                kind = classify_generation(text)
                if kind == "code":
                    st.code_count += 1
                    if matched:
                        st.code_correct += 1
                elif kind == "natural_language":
                    st.nl_count += 1
                    if matched:
                        st.nl_correct += 1
                else:
                    st.unknown_count += 1
                    if matched:
                        st.unknown_correct += 1

    # 3. Custom eval
    if eval_root and eval_root.exists():
        result_files = _collect_custom_eval(eval_root)
        report["sources"].append({"type": "custom_eval", "count": len(result_files)})
        for model_id, task_name, path in result_files:
            rows = _load_custom_eval_results(path)
            if not rows:
                continue
            st = ensure_model_task(model_id, task_name)
            for text, is_correct, _gold in rows:
                kind = classify_generation(text)
                if kind == "code":
                    st.code_count += 1
                    if is_correct:
                        st.code_correct += 1
                elif kind == "natural_language":
                    st.nl_count += 1
                    if is_correct:
                        st.nl_correct += 1
                else:
                    st.unknown_count += 1
                    if is_correct:
                        st.unknown_correct += 1

    # Build report
    for model_id, tasks in all_stats.items():
        report["by_model"][model_id] = {}
        for task, st in tasks.items():
            report["by_model"][model_id][task] = st.to_dict()

    report["summary"] = {
        "total_models": len(all_stats),
        "total_tasks": sum(len(t) for t in all_stats.values()),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="统计每个模型使用代码 vs 自然语言解决问题的数量和准确率",
    )
    ap.add_argument(
        "--lighteval-root",
        type=str,
        default="",
        help="LightEval 输出根目录 (含 details/ 子目录)",
    )
    ap.add_argument(
        "--zeroeval-root",
        type=str,
        default="",
        help="ZeroEval 输出根目录 (含 parsed_*.json)",
    )
    ap.add_argument(
        "--eval-root",
        type=str,
        default="",
        help="Custom eval 输出根目录 (含 results_*.json)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="输出 JSON 文件路径",
    )
    ap.add_argument(
        "--print",
        action="store_true",
        help="打印人类可读摘要到 stdout",
    )
    args = ap.parse_args()

    lighteval_root = Path(args.lighteval_root).resolve() if args.lighteval_root else None
    zeroeval_root = Path(args.zeroeval_root).resolve() if args.zeroeval_root else None
    eval_root = Path(args.eval_root).resolve() if args.eval_root else None

    # Default: use co-learning debug paths (ZeroEval 暂不加入 pipeline，需手动 --zeroeval-root)
    if not any([lighteval_root, zeroeval_root, eval_root]):
        base = PROJECT_ROOT / "debug"
        if (base / "lighteval").exists():
            lighteval_root = base / "lighteval"

    report = _run_stats(
        lighteval_root=lighteval_root,
        zeroeval_root=zeroeval_root,
        eval_root=eval_root,
    )

    out_path = args.output.strip() if args.output else None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {out_path}", file=sys.stderr)

    if args.print or not out_path:
        _print_summary(report)
    return 0


def _print_summary(report: Dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n=== Code vs Natural Language 统计 ===\n")
    by_model = report.get("by_model", {})
    for model_id, tasks in sorted(by_model.items()):
        print(f"Model: {model_id}")
        for task, data in sorted(tasks.items()):
            code = data.get("code", {})
            nl = data.get("natural_language", {})
            print(f"  Task: {task}")
            print(f"    代码: count={code.get('count', 0)}, correct={code.get('correct', 0)}, acc={code.get('accuracy')}")
            print(f"    自然语言: count={nl.get('count', 0)}, correct={nl.get('correct', 0)}, acc={nl.get('accuracy')}")
            print(f"    total={data.get('total', 0)}")
        print()
    print("Sources:", report.get("sources", []))


if __name__ == "__main__":
    sys.exit(main())
