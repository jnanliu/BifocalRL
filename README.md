# Co-Learning

Co-training framework for LLM self-improvement via GRPO / GSPO with self-certainty estimation (SCE), majority voting, and multi-backend evaluation.

**Developer guide (onboarding, two-stack overview, extension points):** [docs/CO_LEARNING_DEVELOPER_GUIDE.md](../docs/CO_LEARNING_DEVELOPER_GUIDE.md)

**Mixed-reward 实验交接说明：** [README_MIXED_REWARD.md](README_MIXED_REWARD.md)

## Directory Structure

```
co-learning/
├── src/
│   ├── training/              # Training entry points
│   │   ├── co_intuitor_train.py      # Co-training (model A + B)
│   │   └── single_intuitor_train.py  # Single-model training
│   ├── pipelines/             # Generate / score / compute pipelines
│   │   ├── generate_and_score.py           # Co-model gen+score
│   │   ├── generate_and_score_single.py    # Single-model gen+score
│   │   ├── generate_and_score_cross.py     # Cross-model gen+score
│   │   ├── generate_only.py               # Generation only
│   │   ├── score_and_compute.py            # Co-model scoring + logps
│   │   └── score_and_compute_single.py     # Single-model scoring + logps
│   └── evaluate_checkpoints.py  # HACRL evaluation CLI
│
├── eval/                      # Evaluation backends (unified interface)
│   ├── base.py                # BaseEvaluator ABC, EvalBackendConfig, EvalResult
│   ├── common/                # Shared scoring primitives
│   │   ├── math_equiv.py     # Math equivalence checking
│   │   ├── gsm8k.py          # GSM8K answer extraction
│   │   ├── cruxeval.py       # CRUXEval data loading / judging
│   │   ├── code_vs_nl.py     # Code vs natural language classification
│   │   ├── math_extraction.py # Shared boxed-answer extraction & simplification
│   │   └── gt_score.py       # Ground-truth scoring
│   ├── hacrl/                 # HACRL evaluation backend
│   │   ├── runner.py          # HACRL eval runner
│   │   ├── scorers.py         # HACRL scorer registry
│   │   └── ...                # Individual scorer modules
│   ├── ttrl/                  # TTRL evaluation backend
│   │   ├── runner.py          # TTRL eval runner
│   │   ├── math.py            # TTRL math grading
│   │   └── gpqa.py            # GPQA scoring
│   ├── lighteval/             # LightEval evaluation backend
│   │   ├── runner.py          # LightEval subprocess wrapper
│   │   ├── custom_tasks/      # Custom task definitions
│   │   └── datasets/          # Local dataset builders
│   └── zeroeval/              # ZeroEval / CRUXEval backend
│       └── runner.py
│
├── utils/                     # Shared utilities
│   ├── per_token_logps.py     # Per-token log probability computation
│   ├── logps.py               # Batch logps from pre-built tensors
│   ├── scoring_helpers.py     # Shared SCE scoring helper
│   ├── cache.py               # Cache for generation / scoring results
│   ├── dataset.py             # Dataset loading
│   ├── generate.py            # vLLM generation
│   ├── score.py               # Self-certainty estimation
│   ├── reward.py              # Reward computation dispatch
│   ├── mv.py                  # Majority voting / WMV
│   ├── model_utils.py         # Model loading helpers
│   ├── tokenization.py        # Tokenization & batch building
│   ├── prompt_utils.py        # Prompt rendering
│   └── ...
│
├── scripts/                   # Shell scripts for launching experiments
│   ├── run_co.sh              # Co-training pipeline
│   ├── run_single.sh          # Single-model training pipeline
│   ├── eval.sh                # Standalone evaluation
│   ├── _lib.sh                # Shared shell functions
│   ├── plotting/              # Visualization scripts
│   └── old/                   # Legacy scripts
│
├── recipes/                   # Experiment configurations (JSON)
│   └── configs/
├── ablation/                  # Ablation experiments
├── third_party/               # Vendored dependencies (ZeroEval)
└── debug/                     # Debug artifacts (git-ignored)
```

## Setup

推荐（conda Python 3.11 + pip + Linux 上 CUDA 12.4 toolkit / nvcc）：见仓库根目录文档 [docs/CO_LEARNING_ENV_SETUP.md](../docs/CO_LEARNING_ENV_SETUP.md)，或从仓库根目录执行：

```bash
bash co-learning/scripts/setup_env.sh
conda activate co-learning
```

仅 pip（已有 Python 3.11 环境时）：

```bash
pip install -r requirements.txt
```

## Running Training

### Single-model GSPO

```bash
bash scripts/run_single.sh recipes/configs/single_intuitor_gspo.json
```

### Co-training

```bash
bash scripts/run_co.sh recipes/configs/co_intuitor_hacpo.json
```

## Running Evaluation

### All backends (from config)

```bash
bash scripts/eval.sh recipes/configs/eval.json --model_path /path/to/checkpoint
```

### Individual backends

```bash
# HACRL
python -m src.evaluate_checkpoints --model_path /path/to/model --datasets math_500,gsm8k

# TTRL
python -m eval.ttrl.runner --model_path /path/to/model

# LightEval
python -m eval.lighteval.runner --model_path /path/to/model

# ZeroEval / CRUXEval
python -m eval.zeroeval.runner --model_path /path/to/model
```

## Evaluation Backend Architecture

Each evaluation backend implements a unified interface (`eval.base.BaseEvaluator`) with independent enable/disable switches in the experiment config:

```json
{
  "eval": {
    "hacrl":    { "enabled": true, ... },
    "ttrl":     { "enabled": false, ... },
    "lighteval": { "enabled": true, ... },
    "zeroeval":  { "enabled": true, ... }
  }
}
```
