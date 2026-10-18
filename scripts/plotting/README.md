# Unified Plotting

Use one fixed script and one JSON config file.

## Run

```bash
python3 co-learning/scripts/plotting/plot_from_json.py
```

Or specify a config:

```bash
python3 co-learning/scripts/plotting/plot_from_json.py \
  --config co-learning/scripts/plotting/plots_config.json
```

Optional output directory override:

```bash
python3 co-learning/scripts/plotting/plot_from_json.py \
  --config co-learning/scripts/plotting/plots_config.json \
  --output-dir co-learning/debug/plots
```

## Mixed-reward eval (TTRL)

1. Place (or symlink) eval trees under `co-learning/debug/data/mixed_reward/eval` so paths match `recipes/configs/mixed_reward` `output.ttrl_dir` suffixes (`co-intuitor-hacpo/q2.5q3`, `single-intuitor-gspo/qwen25_3b`, etc.).
2. Build JSON and render PNGs:

```bash
./co-learning/scripts/plotting/run_mixed_reward_plots.sh
```

Or manually:

```bash
python3 co-learning/scripts/plotting/build_mixed_reward_plots_json.py \
  --eval-root co-learning/debug/data/mixed_reward/eval \
  --output co-learning/scripts/plotting/json/mixed_reward_eval.json

python3 co-learning/scripts/plotting/plot_from_json.py \
  --config co-learning/scripts/plotting/json/mixed_reward_eval.json \
  --output-dir "co-learning/scripts/plotting/output/4.9 mixed_reward"
```

**Figures (per pair × dataset):** dual all pipelines (single+co, both models); single-model all pipelines (A/B); intuitor vs mixed_reward (A/B); mv vs mixed_reward (A/B); intuitor vs mv (A/B); mixed 0.25 vs 0.75 (A/B).

**Batch baselines:** by default, `build_mixed_reward_plots_json.py` reads [`debug/data/mixed_reward/batch_eval/<Model>/mixed_reward/comparison_summary.json`](../../debug/data/mixed_reward/batch_eval) (TTRL pass@1 per `gsm8k` / `math_500`) and draws dotted horizontals: both models on dual-all-pipelines plots, only A or B on single-model plots. Override root with `--batch-eval-root`, or disable with `--no-batch-baseline`.

**Extra baselines (optional):** edit [`json/mixed_reward_baselines.json`](json/mixed_reward_baselines.json). Those lines are appended after batch baselines. Use `"value": null` to skip (`plot_from_json` ignores null). Re-run the builder after editing.

## Notes

### `plot_ttrl_eval_auto.py` notes

- Script path: [`plot_ttrl_eval_auto.py`](plot_ttrl_eval_auto.py)
- This script reads TTRL eval outputs directly from `outputs/.../comparison_summary.json` and does not require hand-written plotting JSON.
- Current default checkpoint range is `start_step=1`, `end_step=15`. For the current single-model mixed-reward runs, do not expect `step=16`.
- Missing `a_checkpoint-<step>/comparison_summary.json` is treated as a non-fatal warning. The script keeps running and the corresponding curve point becomes empty, so the line breaks at that step.

### `plot_ttrl_eval_auto_new.py` data sources

The default mode reads the raw evaluation files under `outputs/mixed_reward/eval` and
also writes one auditable `plot_data.json` under each model's plotting output directory:

```bash
python scripts/plotting/plot_ttrl_eval_auto_new.py
```

To redraw without reading the raw evaluation outputs, use the persisted JSON mode:

```bash
python scripts/plotting/plot_ttrl_eval_auto_new.py \
  --source-mode plot-data
```

This scans `scripts/plotting/output/auto_new/*/plot_data.json` by default. A single
JSON file or another root can be supplied explicitly; `--output-dir` keeps re-plotted
images separate from the original images:

```bash
python scripts/plotting/plot_ttrl_eval_auto_new.py \
  --source-mode plot-data \
  --plot-data-root scripts/plotting/output/auto_new \
  --output-dir scripts/plotting/output/replot
```

The JSON mode uses the saved steps, methods, values, baselines, and line styles. It
does not access or rewrite the raw evaluation results or `plot_data.json` files.

### Step-0 baseline format mismatch

- `a_checkpoint-1`, `a_checkpoint-2`, ... usually store values under:
  - `results["checkpoint-<step>"][dataset]`
- But `a_checkpoint-0/comparison_summary.json` currently uses a different shape:
  - `results["<model_name>"][dataset]`
  - for example `results["Qwen2.5-3B"]["math_500"]`
- `plot_ttrl_eval_auto.py` now supports both formats:
  - it first tries `results["checkpoint-0"][dataset]`
  - if that key is absent and there is exactly one dict-like result entry, it falls back to that single model key

### Shared step-0 baseline behavior

- The auto TTRL plotting script now uses one shared `step0 baseline` per dataset, not one baseline per method.
- It first loads every method's `step0` value, then compares them.
- If all methods agree, that shared value is drawn as the single horizontal baseline.
- If the `step0` values disagree, the script emits a clear warning with:
  - each distinct baseline value
  - how many methods produced it
  - which method labels belong to that value
- In the mismatch case, the script currently chooses the shared baseline by majority vote and draws only that one line.

### Important observed warning

- For the current `llama_3b` plots, `step0` is not actually identical across methods.
- This is now surfaced by the plotting script as a warning instead of being silently hidden.
- If those differences are unexpected, check how each method's `a_checkpoint-0` eval was produced before using the shared baseline as a scientific comparison reference.

## Config model

Single JSON contains all plots:

- `global.default_dpi`: default dpi for all plots
- `plots`: list of plot definitions
- each plot:
  - `title`
  - `output` (file path; relative path is resolved from config file folder)
  - `figure.size` `[width, height]`
  - `x` (`values`, `labels`, `label`, `tick_rotation`)
  - `y` (`label`, `limits`, optional `autoscale_tight`, `clip_01`, `autoscale_pad_relative`, `autoscale_pad_absolute`)
  - `series`: each line with `label`, `y`, optional `x`, `style`, optional `annotate`
  - `baselines`: optional horizontal lines
  - `grid`, `legend`: optional

Edit data directly in JSON and keep script unchanged.
