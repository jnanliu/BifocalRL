#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Mixed-reward experiment runner
#
# Usage:
#   cd /home/chenzj/czj/INTUITOR/beifen/test/co-learning
#   source /opt/conda/etc/profile.d/conda.sh
#   conda activate /home/chenzj/.conda/envs/newm
#
# Then run any command below.
# ═══════════════════════════════════════════════════════════════════════════

CFG=recipes/configs/mixed_reward

# ──────────────────────────────────────────────────────────────────────────
# 0. Ground-truth reward baselines (TTRL-style math_lighteval + hacrl_math_boxed)
#    Co: repro-hacpo (two-model HACPO-style co pipeline). Single: repro-gspo.
# ──────────────────────────────────────────────────────────────────────────

bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5smol/co_repro_hacpo.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5q3/co_repro_hacpo.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q1.7q3/co_repro_hacpo.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5phi4/co_repro_hacpo.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q3smol/co_repro_hacpo.json

bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen2.5_3B/single_repro_gspo.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/SmolLM3_3B/single_repro_gspo.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_4B/single_repro_gspo.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_1.7B/single_repro_gspo.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Phi4/single_repro_gspo.json

# ──────────────────────────────────────────────────────────────────────────
# 1. Single-model baselines (intuitor / mv / mixed_reward)
#    Models: Qwen2.5-3B, SmolLM3-3B, Qwen3-4B, Qwen3-1.7B, Phi-4-mini
#    Mixed reward: 0.25 / 0.75 / 0.5 fixed, plus adaptive per-completion blending.
# ──────────────────────────────────────────────────────────────────────────

# --- Qwen2.5-3B ---
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen2.5_3B/single_intuitor.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen2.5_3B/single_mv.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen2.5_3B/single_mixed_reward_alpha025.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen2.5_3B/single_mixed_reward_alpha075.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen2.5_3B/single_mixed_reward_alpha050.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen2.5_3B/single_mixed_reward_adaptive.json

# --- SmolLM3-3B ---
bash scripts/run_single.sh recipes/configs/mixed_reward/single/SmolLM3_3B/single_intuitor.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/SmolLM3_3B/single_mv.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/SmolLM3_3B/single_mixed_reward_alpha025.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/SmolLM3_3B/single_mixed_reward_alpha075.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/SmolLM3_3B/single_mixed_reward_alpha050.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/SmolLM3_3B/single_mixed_reward_adaptive.json

# --- Qwen3-4B ---
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_4B/single_intuitor.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_4B/single_mv.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_4B/single_mixed_reward_alpha025.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_4B/single_mixed_reward_alpha075.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_4B/single_mixed_reward_alpha050.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_4B/single_mixed_reward_adaptive.json

# --- Qwen3-1.7B ---
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_1.7B/single_intuitor.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_1.7B/single_mv.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_1.7B/single_mixed_reward_alpha025.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_1.7B/single_mixed_reward_alpha075.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_1.7B/single_mixed_reward_alpha050.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Qwen3_1.7B/single_mixed_reward_adaptive.json

# --- Phi-4-mini ---
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Phi4/single_intuitor.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Phi4/single_mv.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Phi4/single_mixed_reward_alpha025.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Phi4/single_mixed_reward_alpha075.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Phi4/single_mixed_reward_alpha050.json
bash scripts/run_single.sh recipes/configs/mixed_reward/single/Phi4/single_mixed_reward_adaptive.json

# ──────────────────────────────────────────────────────────────────────────
# 2. Co-learning: co-intuitor (sum_scores with min-max normalization)
# ──────────────────────────────────────────────────────────────────────────

bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5smol/co_intuitor.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5q3/co_intuitor.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q1.7q3/co_intuitor.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5phi4/co_intuitor.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q3smol/co_intuitor.json

# ──────────────────────────────────────────────────────────────────────────
# 3. Co-learning: co-mixed-reward (alpha=0.25: MV 25% + Intuitor 75%)
# ──────────────────────────────────────────────────────────────────────────

bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5smol/co_mixed_reward_alpha025.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5q3/co_mixed_reward_alpha025.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q1.7q3/co_mixed_reward_alpha025.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5phi4/co_mixed_reward_alpha025.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q3smol/co_mixed_reward_alpha025.json

# ──────────────────────────────────────────────────────────────────────────
# 3b. Co-learning: co-mixed-reward (adaptive per-sample MV/Intuitor weights)
# ──────────────────────────────────────────────────────────────────────────

bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5smol/co_mixed_reward_adaptive.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5q3/co_mixed_reward_adaptive.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q1.7q3/co_mixed_reward_adaptive.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5phi4/co_mixed_reward_adaptive.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q3smol/co_mixed_reward_adaptive.json

# ──────────────────────────────────────────────────────────────────────────
# 3c. Co-learning: co-mixed-reward (alpha=0.5 fixed)
# ──────────────────────────────────────────────────────────────────────────

bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5smol/co_mixed_reward_alpha050.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5q3/co_mixed_reward_alpha050.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q1.7q3/co_mixed_reward_alpha050.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5phi4/co_mixed_reward_alpha050.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q3smol/co_mixed_reward_alpha050.json

# ──────────────────────────────────────────────────────────────────────────
# 4. Co-learning: co-mixed-reward (alpha=0.75: MV 75% + Intuitor 25%)
# ──────────────────────────────────────────────────────────────────────────

bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5smol/co_mixed_reward_alpha075.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5q3/co_mixed_reward_alpha075.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q1.7q3/co_mixed_reward_alpha075.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5phi4/co_mixed_reward_alpha075.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q3smol/co_mixed_reward_alpha075.json

# ──────────────────────────────────────────────────────────────────────────
# 5. Co-learning: co-mv (majority voting)
# ──────────────────────────────────────────────────────────────────────────

bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5smol/co_mv.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5q3/co_mv.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q1.7q3/co_mv.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q2.5phi4/co_mv.json
bash scripts/run_co.sh recipes/configs/mixed_reward/co-q3smol/co_mv.json
