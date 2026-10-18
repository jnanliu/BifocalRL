# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class CoIntuitorConfig(TrainingArguments):
    """
    Configuration for Co-Intuitor training.

    Extends TrainingArguments directly (no trl dependency) with the
    GRPO-specific fields that co-learning actually uses, plus all
    Co-Intuitor-specific parameters.
    """

    # ── GRPO fields (previously inherited from trl.GRPOConfig) ──
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of generations per prompt to sample."},
    )
    max_completion_length: int = field(
        default=2048,
        metadata={"help": "Maximum completion length (tokens). Project default: 2048."},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum prompt length (tokens). None = no truncation."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling threshold."},
    )
    beta: float = field(
        default=0.0,
        metadata={"help": "KL coefficient. 0.0 = no reference model needed."},
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (mu in GRPO)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping (lower bound)."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={"help": "Upper-bound epsilon for clipping. Defaults to epsilon if None."},
    )
    loss_type: str = field(
        default="grpo",
        metadata={"help": "Loss type. Options: grpo, bnpo, dr_grpo."},
    )
    scale_rewards: bool = field(
        default=True,
        metadata={"help": "Whether to scale rewards by group std (GRPO normalization)."},
    )
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "Keyword arguments passed to AutoModelForCausalLM.from_pretrained()."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Whether to remove unused columns in the dataset."},
    )

    # ── Co-Intuitor-specific parameters ──
    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    
    num_generations_per_model: int = field(
        default=4,
        metadata={"help": "Number of responses each model generates per prompt."},
    )
    reward_method: str = field(
        default="sum_scores",
        metadata={
            "help": "Reward method for training. "
            "Options: 'sum_scores' (weight_a*scores_a + weight_b*scores_b), "
            "'self_scores' (each model's completions rewarded only by that model's own score), "
            "'truth' (ground-truth 0/1 rewards from dataset answers), "
            "'co_wmv' (Z-score normalize A/B, combine, WMV pseudo-label -> binary rewards), "
            "'co_mv' (unweighted majority voting, each answer = 1 vote -> binary rewards), "
            "'mixed_reward' (w_mv*co_mv + w_int*norm(sum_scores-style SCE), see mixed_reward_* weights)."
        },
    )
    reward_weight_a: float = field(
        default=1.0,
        metadata={
            "help": "Weight for model A's SCE when reward_method=sum_scores or mixed_reward (intuitor term)."
        },
    )
    reward_weight_b: float = field(
        default=1.0,
        metadata={
            "help": "Weight for model B's SCE when reward_method=sum_scores or mixed_reward (intuitor term)."
        },
    )
    mixed_reward_alpha: float = field(
        default=0.5,
        metadata={
            "help": (
                "Alpha for mixed_reward: MV weight = alpha, Intuitor weight = 1-alpha. "
                "E.g. alpha=0.25 means 25%% MV + 75%% Intuitor."
            )
        },
    )
    mixed_reward_alpha_mode: str = field(
        default="fixed",
        metadata={
            "help": (
                "How to mix MV vs Intuitor advantages for mixed_reward (co-training and single-model). "
                "'fixed': use mixed_reward_alpha globally. "
                "'adaptive': per-element alpha = |A_MV| / (|A_MV| + |A_Int| + eps) (legacy). "
                "'adaptive_sc': alpha = self-consistency — fraction of that model's completions "
                "matching the MV pseudo-label (single-model: mean over G; co-training: per-model "
                "mean over own N completions, broadcast within each block). "
                "'adaptive_sc_zscore': single-model only; batch-normalize the per-prompt "
                "self-consistency scores, map to N(mu, sigma), then clip to [lower, upper] "
                "before blending MV and Intuitor. "
                "'reliability_sc_corr': single-model only; combine batch-standardized "
                "self-consistency and signed advantage correlation into prompt reliability, "
                "then derive advantage and post-normalize sample weights. "
                "'prism': single-model only; add MV and Intuitor advantages with unit weights. "
                "'ettrl': single-model only; multiply the MV advantage by within-prompt "
                "relative self-certainty."
            )
        },
    )
    mixed_reward_final_normalize: bool = field(
        default=True,
        metadata={
            "help": (
                "Apply a batch-level z-score to the combined mixed-reward advantages. "
                "Defaults to true for legacy compatibility; prism and ettrl require false."
            )
        },
    )
    mixed_reward_alpha_zscore_mu: float = field(
        default=0.65,
        metadata={
            "help": (
                "Target mean used by mixed_reward adaptive_sc_zscore when mapping per-prompt "
                "self-consistency scores after z-score normalization."
            )
        },
    )
    mixed_reward_alpha_zscore_sigma: float = field(
        default=0.2,
        metadata={
            "help": (
                "Target std used by mixed_reward adaptive_sc_zscore when mapping per-prompt "
                "self-consistency scores after z-score normalization."
            )
        },
    )
    mixed_reward_alpha_zscore_lower: float = field(
        default=0.0,
        metadata={
            "help": (
                "Explicit lower clipping bound for mixed_reward adaptive_sc_zscore alpha."
            )
        },
    )
    mixed_reward_alpha_zscore_upper: float = field(
        default=1.0,
        metadata={
            "help": (
                "Explicit upper clipping bound for mixed_reward adaptive_sc_zscore alpha."
            )
        },
    )
    mixed_reward_corr_weight_mode: str = field(
        default="none",
        metadata={
            "help": (
                "Optional post-combine prompt weighting for mixed_reward. "
                "'none': disable correlation weighting. "
                "'pearson_threshold_power': compute prompt-level Pearson correlation between "
                "MV and Intuitor advantages, map it to a prompt weight, and multiply the final "
                "combined advantages after final_normalize."
            )
        },
    )
    mixed_reward_corr_weight_threshold: float = field(
        default=0.1,
        metadata={
            "help": (
                "Threshold delta for mixed_reward correlation weighting. "
                "Prompt correlation values below this threshold get zero weight."
            )
        },
    )
    mixed_reward_corr_weight_exponent: float = field(
        default=0.8,
        metadata={
            "help": (
                "Exponent used by mixed_reward Pearson-threshold-power prompt weighting."
            )
        },
    )
    mixed_reward_reliability_sc_weight: float = field(
        default=0.6,
        metadata={"help": "SC coefficient in reliability_sc_corr (not sum-normalized)."},
    )
    mixed_reward_reliability_corr_weight: float = field(
        default=0.4,
        metadata={"help": "Signed Pearson-correlation coefficient in reliability_sc_corr."},
    )
    mixed_reward_reliability_use_sample_weight: bool = field(
        default=True,
        metadata={"help": "Apply the reliability-derived prompt weight after final normalization."},
    )
    mixed_reward_reliability_use_advantage_weight: bool = field(
        default=True,
        metadata={"help": "Use reliability-derived MV/Intuitor advantage mixing weights."},
    )
    mixed_reward_reliability_sample_weight_mu: float = field(
        default=1.0,
        metadata={"help": "Target mean for reliability-derived sample weights."},
    )
    mixed_reward_reliability_sample_weight_sigma: float = field(
        default=0.2,
        metadata={"help": "Target std for reliability-derived sample weights."},
    )
    mixed_reward_reliability_sample_weight_lower: float = field(
        default=0.5,
        metadata={"help": "Lower bound for reliability-derived sample weights."},
    )
    mixed_reward_reliability_sample_weight_upper: float = field(
        default=1.5,
        metadata={"help": "Upper bound for reliability-derived sample weights."},
    )
    mixed_reward_reliability_advantage_weight_mu: float = field(
        default=0.5,
        metadata={"help": "Target mean for reliability-derived MV advantage weights."},
    )
    mixed_reward_reliability_advantage_weight_sigma: float = field(
        default=0.2,
        metadata={"help": "Target std for reliability-derived MV advantage weights."},
    )
    mixed_reward_reliability_advantage_weight_lower: float = field(
        default=0.2,
        metadata={"help": "Lower bound for reliability-derived MV advantage weights."},
    )
    mixed_reward_reliability_advantage_weight_upper: float = field(
        default=0.8,
        metadata={"help": "Upper bound for reliability-derived MV advantage weights."},
    )
    sum_scores_minmax: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, min-max normalize the weighted SCE aggregate "
                "(reward_weight_a*scores_a + reward_weight_b*scores_b) per prompt over 2N completions to [0,1]. "
                "Used for reward_method=sum_scores and for the Intuitor term in mixed_reward "
                "(MV term still uses its own min-max blend in two-phase mixed_reward). "
                "Default False: raw weighted SCE sum for the intuitor signal."
            )
        },
    )
    objective_mode: str = field(
        default="grpo",
        metadata={
            "help": (
                "Policy objective mode. "
                "Options: 'grpo' (token-level clipped ratio) or 'gspo' (sequence-level clipped ratio)."
            )
        },
    )
    co_update_mode: str = field(
        default="mixed",
        metadata={
            "help": (
                "How co-training data is consumed. "
                "Options: 'mixed' (legacy own+other mixed in one pass) or "
                "'two_phase' (self phase, then peer phase)."
            )
        },
    )
    is_length_norm: str = field(
        default="mean",
        metadata={
            "help": (
                "Length normalization strategy for sequence-level IS in GSPO. "
                "Options: 'mean' (sum/len), 'sqrt_len' (sum/sqrt(len)), 'none' (sum)."
            )
        },
    )
    is_strict_old_for_peer: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, peer phase under GSPO requires cached old logps from peer model. "
                "Training errors out if old logps are missing."
            )
        },
    )
    capability_weight_mode: str = field(
        default="sc_ratio",
        metadata={
            "help": (
                "Capability ratio mode used to compute w(k,j). "
                "Options: 'sc_ratio' (SC mean ratio) and 'reward_ratio' (mean reward ratio, used for mv/co_mv)."
            )
        },
    )
    capability_ratio_smoothing: str = field(
        default="latest",
        metadata={
            "help": (
                "Smoothing mode for the mean reward used in capability ratio w(k,j). "
                "'latest' uses only the current step's mean reward (default). "
                "'last_k' averages mean rewards over the last K steps (sliding window)."
            )
        },
    )
    capability_ratio_k: int = field(
        default=5,
        metadata={
            "help": (
                "Window size K for 'last_k' capability ratio smoothing. "
                "The smoothed mean is (1/K) * sum of per-step mean rewards over "
                "steps [t-K+1 .. t]. Only used when capability_ratio_smoothing='last_k'. Default: 5."
            )
        },
    )
    capability_normalize_sce: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, normalize raw SCE scores to [0,1] via z-score + sigmoid "
                "before computing the capability ratio w(k,j). Only affects SCE-based "
                "reward methods (sum_scores, self_scores, sc_ratio mode). "
                "Original rewards used for advantage computation are unchanged. Default: False."
            )
        },
    )
    strong_model_peer_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "When > 0, subtract this value from the weak model's capability weight "
                "w(peer->train) during the strong model's peer-loss scaling. "
                "The stronger model is detected automatically per step (higher mean reward). "
                "Does not affect baseline/advantage computation or the weak model's own training. "
                "Clamped to a minimum of 0.1. 0.0 = disabled (default)."
            )
        },
    )
    model_ids: list[str] = field(
        default_factory=lambda: ["a", "b"],
        metadata={"help": "Model identifiers for co-learning. Current implementation expects two ids."},
    )
    selected_indices_a: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated response indices that model A trains on (0-based into "
                "[A_0..A_{N-1}, B_0..B_{N-1}]).  "
                "Examples: '0,1,2,3' (A's own only), '0,1,2,3,4,5,6,7' (all), "
                "'0,1,2,3,4,5,6' (default co-learning).  "
                "Special value 'all' = use every index.  "
                "If set, overrides learn_config for model A."
            )
        },
    )
    selected_indices_b: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated response indices that model B trains on (0-based into "
                "[A_0..A_{N-1}, B_0..B_{N-1}]).  "
                "Examples: '4,5,6,7' (B's own only), '0,1,2,3,4,5,6,7' (all), "
                "'4,5,6,7,0,1,2' (default co-learning).  "
                "Special value 'all' = use every index.  "
                "If set, overrides learn_config for model B."
            )
        },
    )
    learn_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "JSON string specifying per-model learn counts and selection mode. "
                "Format: '{\"a\":{\"own\":4,\"other\":3},\"b\":{\"own\":4,\"other\":3},\"selection\":\"first\"}'. "
                "own = how many of own responses to learn (-1 = all). "
                "other = how many of other model's responses to learn (-1 = all, 0 = none/SL). "
                "selection = 'first' (static, by generation order, DEFAULT), "
                "'top_score' (rank by score_a+score_b, pick highest), or 'random'. "
                "Takes priority over --selected_indices_*."
            )
        },
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use cache for generation and scoring. If True, will check for existing cache files and load them if complete. Cache files are always saved regardless of this setting."},
    )
    generation_batch_size: int = field(
        default=16,
        metadata={"help": "Number of prompts to generate per batch during distributed generation. Smaller batch size reduces memory pressure and avoids frequent offload/load cycles. Default: 16 prompts per batch (64 responses per batch if num_generations_per_model=4)."},
    )
    scoring_batch_size: int = field(
        default=8,
        metadata={"help": "Number of prompts to score per batch during distributed scoring. Smaller batch size reduces memory pressure and avoids OOM errors. Default: 8 prompts per batch (64 responses per batch if num_generations_per_model=4)."},
    )
    logps_batch_size: int = field(
        default=8,
        metadata={"help": "Number of prompts to compute logps per batch during distributed logps computation. Smaller batch size reduces memory pressure. Default: 8 prompts per batch."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": (
                "(Deprecated) Whether to mask truncated completions (completions without EOS). "
                "This repo keeps truncated completions for training/scoring/logps, so this flag is ignored."
            )
        },
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={"help": "Whether to gather DeepSpeed ZeRO-3 parameters for generation. Set to False to avoid gathering overhead if not using ZeRO-3."},
    )

    # Self-update GSPO epsilon (used by both single and co pipelines)
    self_update_epsilon_low: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Lower epsilon for self-update GSPO clipping. "
                "When set, overrides the global --epsilon for self-phase ratio clipping. "
                "clamp(ratio, 1 - eps_low, 1 + eps_high). Default: None (use global epsilon)."
            )
        },
    )
    self_update_epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Upper epsilon for self-update GSPO clipping. "
                "When set, overrides the global --epsilon_high for self-phase ratio clipping. "
                "Default: None (use global epsilon_high, which defaults to epsilon_low)."
            )
        },
    )

    # Peer-update special clipping (co-pipeline only, two sub-toggles)
    peer_special_clip_asymmetric_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable asymmetric clipping for peer-update: the IS ratio can only downweight "
                "(ratio <= 1.0), never upweight. This prevents the model from increasing likelihood "
                "on peer-generated responses."
            )
        },
    )
    peer_special_clip_stepwise_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable stepwise clipping for peer-update: the lower bound of the IS ratio clip "
                "tightens with each optimizer step k within the co-step. "
                "lower = 1 - delta + k * delta_step."
            )
        },
    )
    peer_special_clip_delta: float = field(
        default=0.8,
        metadata={
            "help": (
                "Delta parameter for stepwise peer clipping. "
                "Clip lower bound = 1 - delta + k * delta_step. Default: 0.8."
            )
        },
    )
    peer_special_clip_delta_step: float = field(
        default=0.4,
        metadata={
            "help": (
                "Delta-step parameter for stepwise peer clipping. "
                "Clip lower bound = 1 - delta + k * delta_step. Default: 0.4."
            )
        },
    )

    stable_perf: int = field(
        default=0,
        metadata={
            "help": (
                "Number of initial training steps during which the capability ratio "
                "w(k,j) is forced to 1.0 (i.e. no capability weighting). "
                "Matches HACRL's stable_perf parameter. 0 = disabled (default)."
            )
        },
    )

    peer_is_reweight_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable non-gradient exponential reweighting for peer IS ratios in J_hete. "
                "When enabled and the IS ratio < 1.0, the ratio is reweighted as "
                "ratio * sg(ratio)^alpha to suppress overly aggressive cross-agent "
                "importance weights."
            )
        },
    )
    peer_is_reweight_alpha: float = field(
        default=0.5,
        metadata={
            "help": (
                "Alpha (α ≥ 0) for non-gradient exponential reweighting in J_hete. "
                "Controls the degree of conservativeness: larger alpha suppresses "
                "low IS ratios more aggressively. Only used when "
                "peer_is_reweight_enabled is True. Default: 0.5."
            )
        },
    )

    # Retokenize pipeline switches
    gspo_surrogate_ratio_enabled: bool = field(
        default=True,
        metadata={
            "help": (
                "When True (default), GSPO uses the stopgrad surrogate per-token ratio: "
                "log_ratio = logπ - sg(logπ) + sg(mean_logπ - mean_logπ_old). "
                "When False, GSPO uses a standard per-token ratio like GRPO: "
                "log_ratio = logπ - logπ_old."
            )
        },
    )
    peer_retokenize_old_logps: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, the peer phase uses the training model's own old logps "
                "computed on the peer's retokenized responses (A_old on B_resp) "
                "instead of the peer model's old logps (B_old on B_resp). "
                "Requires cross old logps to be pre-computed by score_and_compute.py."
            )
        },
    )
    peer_loss_uniform: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, the peer loss uses the same PPO-clip formula as self loss: "
                "max(-adv*r, -adv*clip(r)) * cw. The capability weight is kept for "
                "gradient scaling but IS reweight and asymmetric/stepwise clips are "
                "expected to be disabled separately."
            )
        },
    )

    # Cross-step LR scheduling
    total_pipeline_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Total number of outer pipeline steps (generate-score-train iterations). "
                "The single-model pipeline multiplies this by the exact optimizer updates per rollout "
                "to obtain the scheduler's global update count. 0 disables cross-step scheduling."
            )
        },
    )
    current_pipeline_step: int = field(
        default=0,
        metadata={
            "help": (
                "Current outer pipeline step (1-indexed). Used with --total_pipeline_steps "
                "to validate restored optimizer/scheduler state or reconstruct the exact "
                "optimizer-update offset for a stateless legacy resume. 0 = disabled."
            )
        },
    )


@dataclass
class CoIntuitorScriptArguments:
    """
    Script arguments for the Co-Intuitor training script.
    Standalone dataclass (no trl dependency).
    """

    dataset_key: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset key from recipes/datasets.json (e.g., 'math_500', 'gsm8k')."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name from Hugging Face Hub (e.g., 'DigitalLearningGmbH/MATH-lighteval'). Alternative to dataset_key."},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config name (optional, for datasets with configs)."},
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the local dataset directory (alternative to dataset_name or dataset_key)."},
    )
    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Training split name."},
    )
    dataset_test_split: Optional[str] = field(
        default=None,
        metadata={"help": "Test split name (optional)."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use from the dataset. If None, uses all samples."},
    )
    
    # Co-Intuitor-specific arguments
    model_a_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model A (if different from main model)."},
    )
    model_b_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model B (if different from main model)."},
    )
    tokenizer_a_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Tokenizer source-of-truth for model A. If set, we load tokenizer/chat_template from this path "
                "(typically the initial/base model) instead of from the evolving checkpoint directory."
            )
        },
    )
    tokenizer_b_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Tokenizer source-of-truth for model B. If set, we load tokenizer/chat_template from this path "
                "(typically the initial/base model) instead of from the evolving checkpoint directory."
            )
        },
    )
    train_model: Optional[str] = field(
        default="a",
        metadata={"help": "Which model to train: 'a' or 'b'. Determines which model is the policy model."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing cache.json file (generated by generate_and_score.py). If None, uses output_dir."},
    )
