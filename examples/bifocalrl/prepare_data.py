"""Convert a local/Hugging Face dataset to OpenRLHF JSONL prompts."""

import argparse
import json
from pathlib import Path


def convert(
    dataset, output, *, prompt_key, label_key=None, label_format="raw", limit=None,
    plain_prompts=False, datasource=None,
):
    count = 0
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", encoding="utf-8") as stream:
        for row in dataset:
            if limit is not None and count >= limit:
                break
            prompt = row[prompt_key]
            if label_format == "sanbdox_fusion":
                prefix = ""
            else:
                prefix = "Please solve the following problem step by step and put your final answer enclosed in \\boxed{}.\n\n"
            if isinstance(prompt, str) and not plain_prompts:
                prompt = [{"role": "user", "content":  prefix + prompt}]
            record = {
                "prompt": prompt,
                "datasource": datasource if datasource is not None else row.get("datasource") or "default",
            }
            if label_key:
                label = row[label_key]
                if label_format == "gsm8k":
                    label = str(label).split("####")[-1].strip()
                elif label_format == "sandbox_fusion":
                    cases = json.loads(label) if isinstance(label, str) else label
                    label = json.dumps({"backend": "sandbox_fusion", "test_cases": cases}, ensure_ascii=False)
                record["label"] = label
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="HF repo ID, JSON/JSONL/Parquet file, or load_from_disk directory")
    parser.add_argument("output")
    parser.add_argument("--subset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", help="Pin an HF dataset revision for reproducibility")
    parser.add_argument("--prompt-key", default="problem")
    parser.add_argument("--label-key", help="Omit for unlabelled training; supply answer column for evaluation")
    parser.add_argument("--label-format", choices=["raw", "gsm8k", "sandbox_fusion"], default="raw")
    parser.add_argument(
        "--datasource", help="Set the datasource for every row; otherwise preserve the existing value or use 'default'"
    )
    parser.add_argument(
        "--plain-prompts", action="store_true", help="Keep string prompts for base models without a chat template"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.label_format == "sandbox_fusion" and not args.label_key:
        parser.error("--label-format sandbox_fusion requires --label-key with the test cases column")
    from datasets import load_dataset, load_from_disk

    path = Path(args.dataset)
    if path.is_dir():
        dataset = load_from_disk(str(path))
        if hasattr(dataset, "keys"):
            dataset = dataset[args.split]
    elif path.is_file():
        kind = "parquet" if path.suffix == ".parquet" else "json"
        dataset = load_dataset(kind, data_files=str(path), split="train")
    else:
        dataset = load_dataset(args.dataset, args.subset, split=args.split, revision=args.revision)
    count = convert(
        dataset,
        args.output,
        prompt_key=args.prompt_key,
        label_key=args.label_key,
        label_format=args.label_format,
        limit=args.limit,
        plain_prompts=args.plain_prompts,
        datasource=args.datasource,
    )
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
