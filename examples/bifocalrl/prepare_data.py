"""Convert a local/Hugging Face dataset to OpenRLHF JSONL prompts."""

import argparse
import json
from pathlib import Path


def convert(dataset, output, *, prompt_key, label_key=None, label_format="raw", limit=None, plain_prompts=False):
    count = 0
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", encoding="utf-8") as stream:
        for row in dataset:
            if limit is not None and count >= limit:
                break
            prompt = row[prompt_key]
            if isinstance(prompt, str) and not plain_prompts:
                prompt = [{"role": "user", "content": prompt}]
            record = {"prompt": prompt}
            if label_key:
                label = row[label_key]
                if label_format == "gsm8k":
                    label = str(label).split("####")[-1].strip()
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
    parser.add_argument("--label-format", choices=["raw", "gsm8k"], default="raw")
    parser.add_argument(
        "--plain-prompts", action="store_true", help="Keep string prompts for base models without a chat template"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
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
    )
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
