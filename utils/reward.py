# coding=utf-8
"""
Unified reward and scoring module for Co-Intuitor.

Reward computation
------------------
All reward methods ultimately produce per-response scalar rewards.
The :func:`compute_rewards` dispatcher routes to the appropriate backend so
callers never need to branch on ``reward_method``.

Supported reward methods:

- **truth**       : Ground-truth binary reward (0/1) via HACRL scorers.
- **co_mv**       : Joint majority voting across both models' responses.
- **self_mv**     : Independent majority voting per model.
- **sum_scores**  : Cross-model SCE scoring (caller pre-computes SCE).
- **self_scores** : Self-model SCE scoring  (caller pre-computes SCE).
- **co_wmv**      : SCE-weighted majority voting (caller pre-computes SCE;
                    weighting + voting happen at training time).
- **mv_gated_intuitor** : R = R(MV) * sigmoid(R_self(intuitor)).
                    Self-scored variant: each model uses its own SCE as R(intuitor).
                    MV provides a hard binary gate; sigmoid bounds SCE to (0, 1).
- **co_mv_gated_intuitor** : R = R(MV) * sigmoid(R_a(intuitor) + R_b(intuitor)).
                    Cross-scored variant: R(intuitor) is the sum of both models'
                    SCE scores for the same response, then sigmoid-bounded.
- **mixed_reward** : A = w_mv * GRPO(R_mv) + w_int * GRPO(R_int), optionally followed by
                    a batch z-score on the blend.
                    MV and Intuitor rewards are independently GRPO group-normalized
                    (mean-subtract + std-divide per prompt group) before being weighted
                    and combined. Legacy modes z-score the combined tensor over the full
                    batch (all prompts and generations); PRISM/eTTRL explicitly skip that
                    final normalization. Cached scores are raw cross SCE (identical layout
                    to ``sum_scores``).

Self-certainty scoring
----------------------
:func:`score_responses` computes per-response self-certainty advantage scores,
backed by :func:`get_per_token_self_certainty` and :func:`get_advantage_from_sce`.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from accelerate import Accelerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REWARD_METHODS = frozenset({
    "truth", "co_mv", "self_mv", "sum_scores", "self_scores", "co_wmv",
    "mv_gated_intuitor", "co_mv_gated_intuitor", "mixed_reward",
})

SCE_METHODS = frozenset({
    "sum_scores", "self_scores", "co_wmv",
    "mv_gated_intuitor", "co_mv_gated_intuitor", "mixed_reward",
})


# ---------------------------------------------------------------------------
# Reward dispatch
# ---------------------------------------------------------------------------

def compute_rewards(
    reward_method: str,
    responses_a: List[str],
    responses_b: List[str],
    num_prompts: int,
    num_generations: int,
    *,
    ground_truths: Optional[List[str]] = None,
    scorer_name: str = "hacrl_prime_math",
    dataset_hint: Optional[str] = None,
    sce_scores_a: Optional[torch.Tensor] = None,
    sce_scores_b: Optional[torch.Tensor] = None,
    step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rewards for both models using the specified method.

    For non-SCE methods (``truth``, ``co_mv``, ``self_mv``), rewards are
    computed internally from ``responses_a / responses_b``.

    For SCE-based methods (``sum_scores``, ``self_scores``, ``co_wmv``,
    ``mixed_reward``, …),
    the caller must pre-compute SCE advantage tensors via model inference
    and pass them as ``sce_scores_a / sce_scores_b``.  This function
    validates shapes and returns them directly (the actual combination /
    weighting happens at training time).

    Args:
        reward_method: One of :data:`REWARD_METHODS`.
        responses_a: Flattened text responses from Model A
            (length = ``num_prompts * num_generations``).
        responses_b: Flattened text responses from Model B
            (length = ``num_prompts * num_generations``).
        num_prompts: Number of unique prompts.
        num_generations: Generations per model per prompt (*N*).
        ground_truths: Per-prompt ground-truth answers (for ``"truth"``).
        scorer_name: HACRL scorer backend (for ``"truth"``).
        dataset_hint: Dataset identifier forwarded to scorer.
        sce_scores_a: Pre-computed SCE advantages from Model A scorer.
        sce_scores_b: Pre-computed SCE advantages from Model B scorer.
        step: Current training step (seeds RNG for MV tie-breaking).

    Returns:
        ``(rewards_a, rewards_b)`` — each is a 2-D ``torch.Tensor`` whose
        shape is ``(num_prompts, G)`` for per-model methods or
        ``(num_prompts, 2G)`` for cross-model methods.
    """
    if reward_method not in REWARD_METHODS:
        raise ValueError(
            f"Unknown reward_method {reward_method!r}. "
            f"Supported: {sorted(REWARD_METHODS)}"
        )

    N = int(num_generations)

    if reward_method == "truth":
        return _truth_rewards(
            responses_a, responses_b, num_prompts, N,
            ground_truths=ground_truths,
            scorer_name=scorer_name,
            dataset_hint=dataset_hint,
        )

    if reward_method == "co_mv":
        return _co_mv_rewards(responses_a, responses_b, num_prompts, N, step=step)

    if reward_method == "self_mv":
        return _self_mv_rewards(responses_a, responses_b, num_prompts, N, step=step)

    if reward_method in ("mv_gated_intuitor", "co_mv_gated_intuitor"):
        if sce_scores_a is None or sce_scores_b is None:
            raise ValueError(
                f"reward_method={reward_method!r} requires pre-computed SCE scores "
                "(sce_scores_a / sce_scores_b)"
            )
        cross = reward_method == "co_mv_gated_intuitor"
        return _mv_gated_intuitor_rewards(
            responses_a, responses_b, num_prompts, N,
            sce_scores_a=sce_scores_a,
            sce_scores_b=sce_scores_b,
            step=step,
            cross=cross,
        )

    # SCE-based methods — pass through pre-computed scores.
    if sce_scores_a is None or sce_scores_b is None:
        raise ValueError(
            f"reward_method={reward_method!r} requires pre-computed SCE scores "
            "(sce_scores_a / sce_scores_b)"
        )
    return sce_scores_a, sce_scores_b


# ---------------------------------------------------------------------------
# Internal reward helpers
# ---------------------------------------------------------------------------

def _truth_rewards(
    responses_a: List[str],
    responses_b: List[str],
    num_prompts: int,
    N: int,
    *,
    ground_truths: Optional[List[str]] = None,
    scorer_name: str = "hacrl_prime_math",
    dataset_hint: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Binary 0/1 rewards by comparing completions against ground-truth."""
    if ground_truths is None:
        raise ValueError("ground_truths required for reward_method='truth'")

    from eval.common.gt_score import score_completions_against_ground_truth

    total_per_prompt = 2 * N

    combined: List[str] = []
    for i in range(num_prompts):
        sa, ea = i * N, (i + 1) * N
        combined.extend(responses_a[sa:ea])
        combined.extend(responses_b[sa:ea])

    gt_scores = score_completions_against_ground_truth(
        completions=combined,
        ground_truths=ground_truths,
        num_generations=total_per_prompt,
        scorer_name=scorer_name,
        dataset_hint=dataset_hint,
    )

    all_gt = torch.tensor(gt_scores).view(num_prompts, total_per_prompt)
    return all_gt[:, :N], all_gt[:, N:]


def _co_mv_rewards(
    responses_a: List[str],
    responses_b: List[str],
    num_prompts: int,
    N: int,
    *,
    step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Joint majority voting: all 2N responses per prompt share one pseudo-label."""
    from utils.mv import rewards_from_mv_pseudolabel

    ra_list: List[List[float]] = []
    rb_list: List[List[float]] = []
    for i in range(num_prompts):
        sa, ea = i * N, (i + 1) * N
        combined_resp = list(responses_a[sa:ea]) + list(responses_b[sa:ea])
        rng = random.Random((int(step) + 1) * 1_000_000 + i)
        r_all, _ = rewards_from_mv_pseudolabel(combined_resp, rng=rng)
        ra_list.append(r_all[:N])
        rb_list.append(r_all[N:])

    rewards_a = torch.tensor(ra_list)
    rewards_b = torch.tensor(rb_list)

    n_correct = int((rewards_a > 0.5).sum().item() + (rewards_b > 0.5).sum().item())
    n_total = rewards_a.numel() + rewards_b.numel()
    logger.info(
        "co_mv rewards: %d/%d correct (%.1f%%)",
        n_correct, n_total, 100.0 * n_correct / max(n_total, 1),
    )
    return rewards_a, rewards_b


def _self_mv_rewards(
    responses_a: List[str],
    responses_b: List[str],
    num_prompts: int,
    N: int,
    *,
    step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent majority voting: each model votes among its own N responses."""
    from utils.mv import rewards_from_mv_pseudolabel

    ra_list: List[List[float]] = []
    rb_list: List[List[float]] = []
    for i in range(num_prompts):
        sa, ea = i * N, (i + 1) * N
        rng_a = random.Random((int(step) + 1) * 1_000_000 + i)
        rng_b = random.Random((int(step) + 1) * 2_000_000 + i)
        r_a, _ = rewards_from_mv_pseudolabel(responses_a[sa:ea], rng=rng_a)
        r_b, _ = rewards_from_mv_pseudolabel(responses_b[sa:ea], rng=rng_b)
        ra_list.append(r_a)
        rb_list.append(r_b)

    rewards_a = torch.tensor(ra_list)
    rewards_b = torch.tensor(rb_list)

    n_correct = int((rewards_a > 0.5).sum().item() + (rewards_b > 0.5).sum().item())
    n_total = rewards_a.numel() + rewards_b.numel()
    logger.info(
        "self_mv rewards: %d/%d correct (%.1f%%)",
        n_correct, n_total, 100.0 * n_correct / max(n_total, 1),
    )
    return rewards_a, rewards_b


def _mv_gated_intuitor_rewards(
    responses_a: List[str],
    responses_b: List[str],
    num_prompts: int,
    N: int,
    *,
    sce_scores_a: torch.Tensor,
    sce_scores_b: torch.Tensor,
    step: int = 0,
    cross: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MV-gated intuitor: R = R(MV) * sigmoid(R(intuitor)).

    MV provides a hard binary gate — only responses whose extracted answer
    matches the majority-vote pseudo-label get a non-zero reward.  The
    intuitor (SCE) score is passed through sigmoid to bound it in (0, 1),
    preventing extreme values from dominating the advantage computation.

    Two variants controlled by *cross*:

    - ``cross=False`` (**self**, ``mv_gated_intuitor``): each model uses only
      its own SCE score.  ``sce_scores_a[:, :N]`` for A's responses,
      ``sce_scores_b[:, N:]`` for B's responses (or ``[:, :N]`` when
      ``score_only`` was used and shapes are ``(P, N)``).
    - ``cross=True`` (**co**, ``co_mv_gated_intuitor``): both models score
      every response; ``R(intuitor) = SCE_a + SCE_b`` for the same response.
      Requires ``sce_scores_a/b`` to have shape ``(P, 2N)`` — the first N
      columns correspond to Model A's responses, the last N to Model B's.
    """
    from utils.mv import rewards_from_mv_pseudolabel

    mv_a_list: List[List[float]] = []
    mv_b_list: List[List[float]] = []
    for i in range(num_prompts):
        sa, ea = i * N, (i + 1) * N
        combined_resp = list(responses_a[sa:ea]) + list(responses_b[sa:ea])
        rng = random.Random((int(step) + 1) * 1_000_000 + i)
        r_all, _ = rewards_from_mv_pseudolabel(combined_resp, rng=rng)
        mv_a_list.append(r_all[:N])
        mv_b_list.append(r_all[N:])

    mv_a = torch.tensor(mv_a_list)  # (P, N) binary 0/1
    mv_b = torch.tensor(mv_b_list)  # (P, N) binary 0/1

    if cross:
        # Both models score all 2N responses: shape (P, 2N).
        # A's responses = [:, :N], B's responses = [:, N:].
        combined_sce_a = sce_scores_a[:, :N] + sce_scores_b[:, :N]
        combined_sce_b = sce_scores_a[:, N:] + sce_scores_b[:, N:]
        sig_sce_a = torch.sigmoid(combined_sce_a)
        sig_sce_b = torch.sigmoid(combined_sce_b)
    else:
        # Self-scored: each model uses its own SCE for its own responses.
        # When score_only was used, shapes are (P, N); otherwise (P, 2N)
        # and Model A's self-score = sce_scores_a[:, :N],
        #     Model B's self-score = sce_scores_b[:, N:] (or [:, :N] if (P, N)).
        sig_sce_a = torch.sigmoid(sce_scores_a[:, :N])
        b_cols = sce_scores_b.shape[1]
        sig_sce_b = torch.sigmoid(
            sce_scores_b[:, N:] if b_cols > N else sce_scores_b[:, :N]
        )

    rewards_a = mv_a * sig_sce_a
    rewards_b = mv_b * sig_sce_b

    tag = "co_mv_gated_intuitor" if cross else "mv_gated_intuitor"
    n_mv = int((mv_a > 0.5).sum().item() + (mv_b > 0.5).sum().item())
    n_total = mv_a.numel() + mv_b.numel()
    logger.info(
        "%s rewards: MV match %d/%d (%.1f%%), gated reward mean=%.4f",
        tag, n_mv, n_total, 100.0 * n_mv / max(n_total, 1),
        (rewards_a.mean().item() + rewards_b.mean().item()) / 2,
    )
    return rewards_a, rewards_b


# ---------------------------------------------------------------------------
# Self-certainty scoring
# ---------------------------------------------------------------------------

from utils.tokenization import build_left_padded_prompt_completion_batch


def get_per_token_self_certainty(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Computes per-token self-certainty in batches.

    Args:
        model: Model to compute self-certainty with
        input_ids: Input token IDs (B, L)
        attention_mask: Attention mask (B, L)
        logits_to_keep: Number of logit positions to keep
        batch_size: Batch size for processing (if None, uses input_ids.size(0))

    Returns:
        Per-token self-certainty scores (B, logits_to_keep)
    """
    batch_size = batch_size or input_ids.size(0)
    all_sce = []

    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = input_ids.device

    num_batches = (input_ids.size(0) + batch_size - 1) // batch_size
    for i in range(0, input_ids.size(0), batch_size):
        input_ids_batch = input_ids[i : i + batch_size]
        attention_mask_batch = attention_mask[i : i + batch_size]

        if model_device is not None:
            input_ids_batch = input_ids_batch.to(model_device, non_blocking=True)
            attention_mask_batch = attention_mask_batch.to(model_device, non_blocking=True)

        logits = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
        sce_chunk = logits[:, -logits_to_keep:, :]  # (B, L_keep, V)
        sce_values = torch.logsumexp(sce_chunk, dim=-1) - sce_chunk.mean(dim=-1)
        all_sce.append(sce_values)

    return torch.cat(all_sce, dim=0)


@torch.no_grad()
def get_advantage_from_sce(
    SCe: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates advantage scores from SCe values and returns a 1D tensor with a single advantage per sample.

    Args:
        SCe: (B, L) tensor - Per-token score
        completion_mask: (B, L) bool / 0-1 tensor - 1 for valid tokens, 0 for padding

    Returns:
        advantage: (B,) tensor - Single advantage per sample
    """
    output_lengths = completion_mask.sum(dim=1, dtype=torch.long)  # (B,)
    advantage = torch.zeros(SCe.size(0), device=SCe.device, dtype=SCe.dtype)
    valid_mask = output_lengths > 0
    if valid_mask.any():
        sce_sum = (SCe * completion_mask).sum(dim=1)
        advantage[valid_mask] = sce_sum[valid_mask] / output_lengths[valid_mask].to(SCe.dtype)
    return advantage.detach()


def score_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    responses: List[str],
    accelerator: Optional["Accelerator"] = None,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    mask_truncated_completions: bool = False,
    completion_is_truncated: Optional[List[Optional[bool]]] = None,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    gather_deepspeed3_params: bool = True,
) -> List[float]:
    """
    Score responses using self-certainty.

    Args:
        model: Model to use for scoring
        tokenizer: Tokenizer for the model
        prompts: List of prompt texts (should match responses in structure)
        responses: List of response texts (flattened: all responses for all prompts)
        accelerator: Optional accelerator for distributed training
        max_prompt_length: Maximum prompt length
        batch_size: Batch size for scoring
        device: Device to use
        gather_deepspeed3_params: Whether to gather DeepSpeed ZeRO-3 parameters

    Returns:
        List of advantage scores (same length as responses)
    """
    if device is None:
        if accelerator is not None:
            device = accelerator.device
        else:
            device = next(model.parameters()).device

    model.eval()

    build_device = torch.device("cpu")
    input_ids, attention_mask, completion_mask_window, logits_to_keep = (
        build_left_padded_prompt_completion_batch(
            tokenizer,
            prompts,
            responses,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            mask_truncated_completions=mask_truncated_completions,
            completion_is_truncated=completion_is_truncated,
            device=build_device,
        )
    )

    zero_stage_3 = False
    if accelerator is not None:
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

    with torch.no_grad():
        if zero_stage_3 and gather_deepspeed3_params:
            from utils.trl_compat import unwrap_model_for_generation
            with unwrap_model_for_generation(
                model, accelerator, gather_deepspeed3_params=True
            ) as unwrapped_model:
                sce_per_token = get_per_token_self_certainty(
                    unwrapped_model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size
                )
        else:
            sce_per_token = get_per_token_self_certainty(
                model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size
            )

    completion_mask_window = completion_mask_window.to(sce_per_token.device, non_blocking=True)
    advantages = get_advantage_from_sce(sce_per_token, completion_mask_window)

    return advantages.cpu().tolist()
