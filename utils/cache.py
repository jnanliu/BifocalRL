# coding=utf-8
"""
Cache module for Co-Intuitor.

This module handles caching of generations, scores, and logps.
IMPORTANT: Cache only stores the LATEST step data. Previous steps are overwritten.

Cache format (cache.json):
{
    "generations_a": {
        "step": int,  # Latest step number
        "total_unique_prompts": int,
        "text_responses": List[str],
        "prompts": Optional[List[str]]  # Cached prompts for this step
    },
    "generations_b": {
        "step": int,  # Latest step number
        "total_unique_prompts": int,
        "text_responses": List[str],
        "prompts": Optional[List[str]]  # Cached prompts for this step
    },
    "scores_a": {
        "step": int,  # Latest step number
        "advantages": List[float]
    },
    "scores_b": {
        "step": int,  # Latest step number
        "advantages": List[float]
    },
    "checkpoint_info": {
        "step": int,  # Latest step number
        "start_index": int,
        "sampled_indices": List[int]
    },
    # Optional append-only histories (training / diagnostics; not overwritten like latest-step keys):
    # "mean_rewards_history", "mean_sc_scores_history",
    # "reward_stats_history": list of {"step", "reward_stats": {model_id: {"mean", "var"}}},
    # "capability_ratio_history": list of {"step", ... raw_ratios, clipped_ratios, applied_weights, ...}
}

Logps cache format (logps_cache.json):
{
    "logps_a": {
        "step": int,  # Latest step number
        "old_per_token_logps": List[List[float]],  # (num_samples, completion_length)
        "ref_per_token_logps": Optional[List[List[float]]]  # None if beta == 0
    },
    "logps_b": {
        "step": int,  # Latest step number
        "old_per_token_logps": List[List[float]],
        "ref_per_token_logps": Optional[List[List[float]]]
    }
}
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class CoIntuitorCache:
    """Cache manager for Co-Intuitor generations, scores, and logps."""
    
    def __init__(self, cache_file: str):
        """
        Initialize cache manager.
        
        Args:
            cache_file: Path to cache JSON file (generations and scores)
        """
        self.cache_file = cache_file
        # Create logps cache file path: replace cache.json with logps_cache.json
        if cache_file.endswith("cache.json"):
            self.logps_cache_file = cache_file.replace("cache.json", "logps_cache.json")
        else:
            # If cache_file doesn't end with cache.json, append logps_cache.json
            cache_dir = os.path.dirname(cache_file) if os.path.dirname(cache_file) else "."
            self.logps_cache_file = os.path.join(cache_dir, "logps_cache.json")
        self._cache_data = None
        self._logps_cache_data = None
        self._load_cache()
        self._load_logps_cache()
    
    def _load_cache(self):
        """Load cache from file with retry for transient filesystem issues."""
        import time
        for attempt in range(3):
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        self._cache_data = json.load(f)
                    return
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    logger.warning(f"Failed to load cache file after retries: {e}")
                    self._cache_data = {}
            else:
                self._cache_data = {}
                return
    
    @staticmethod
    def _get_rank() -> int:
        for var in ("RANK", "LOCAL_RANK"):
            v = os.environ.get(var)
            if v is not None and str(v).strip().isdigit():
                return int(v)
        return 0

    def _save_cache(self):
        """Save cache to file.

        Only rank-0 writes to avoid multi-process races on the same tmp file.
        Other ranks update their in-memory ``_cache_data`` but skip the disk I/O;
        the next ``_load_cache()`` call will pick up rank-0's write.
        """
        if self._get_rank() != 0:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            tmp_path = f"{self.cache_file}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache_data, f, indent=2, ensure_ascii=False)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, self.cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")
    
    def _load_logps_cache(self):
        """Load logps cache from file."""
        if os.path.exists(self.logps_cache_file):
            try:
                with open(self.logps_cache_file, "r", encoding="utf-8") as f:
                    self._logps_cache_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load logps cache file: {e}")
                self._logps_cache_data = {}
        else:
            self._logps_cache_data = {}
    
    def _save_logps_cache(self):
        """Save logps cache to file (rank-0 only)."""
        if self._get_rank() != 0:
            return
        try:
            os.makedirs(os.path.dirname(self.logps_cache_file), exist_ok=True)
            tmp_path = f"{self.logps_cache_file}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._logps_cache_data, f, indent=2, ensure_ascii=False)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, self.logps_cache_file)
        except Exception as e:
            logger.warning(f"Failed to save logps cache file: {e}")
    
    def save_generations_a(
        self,
        step: int,
        total_unique_prompts: int,
        text_responses: List[str],
        prompts: Optional[List[str]] = None,
        finish_reasons: Optional[List[Optional[str]]] = None,
    ):
        """Save Model A generations to cache."""
        gen_data = {
            "step": step,
            "total_unique_prompts": total_unique_prompts,
            "text_responses": text_responses,
        }
        if prompts is not None:
            gen_data["prompts"] = prompts
        if finish_reasons is not None:
            gen_data["finish_reasons"] = finish_reasons
        self._cache_data["generations_a"] = gen_data
        self._save_cache()
    
    def save_generations_b(
        self,
        step: int,
        total_unique_prompts: int,
        text_responses: List[str],
        prompts: Optional[List[str]] = None,
        finish_reasons: Optional[List[Optional[str]]] = None,
    ):
        """Save Model B generations to cache."""
        gen_data = {
            "step": step,
            "total_unique_prompts": total_unique_prompts,
            "text_responses": text_responses,
        }
        if prompts is not None:
            gen_data["prompts"] = prompts
        if finish_reasons is not None:
            gen_data["finish_reasons"] = finish_reasons
        self._cache_data["generations_b"] = gen_data
        self._save_cache()
    
    @staticmethod
    def _flatten_scores(advantages) -> List[float]:
        """Flatten potentially nested score lists (e.g. from 2D tensor.tolist())."""
        if advantages and isinstance(advantages[0], (list, tuple)):
            return [x for sublist in advantages for x in sublist]
        return advantages

    def save_scores_a(
        self,
        step: int,
        advantages: List[float],
    ):
        """Save Model A scores to cache."""
        self._cache_data["scores_a"] = {
            "step": step,
            "advantages": self._flatten_scores(advantages),
        }
        self._save_cache()
    
    def save_scores_b(
        self,
        step: int,
        advantages: List[float],
    ):
        """Save Model B scores to cache."""
        self._cache_data["scores_b"] = {
            "step": step,
            "advantages": self._flatten_scores(advantages),
        }
        self._save_cache()
    
    def load_generations_a(
        self, step: int, num_generations_per_model: int
    ) -> Tuple[bool, Optional[List[str]], Optional[int], Optional[List[str]], Optional[List[Optional[str]]]]:
        """
        Load Model A generations from cache.
        
        Args:
            step: Expected step number
            num_generations_per_model: Number of generations per prompt
        
        Returns:
            Tuple of (is_valid, text_responses, total_unique_prompts, cached_prompts, finish_reasons)
        """
        # Reload from disk to support multi-process usage (main rank writes; others read).
        self._load_cache()
        if "generations_a" not in self._cache_data:
            return False, None, None, None, None
        
        gen_data = self._cache_data["generations_a"]
        cache_step = gen_data.get("step")
        cache_responses = gen_data.get("text_responses", [])
        cache_total_prompts = gen_data.get("total_unique_prompts", 0)
        cached_prompts = gen_data.get("prompts", None)
        cached_finish = gen_data.get("finish_reasons", None)
        
        expected_count = cache_total_prompts * num_generations_per_model
        
        if cache_step == step and len(cache_responses) == expected_count:
            if cached_finish is not None and len(cached_finish) != expected_count:
                cached_finish = None
            return True, cache_responses, cache_total_prompts, cached_prompts, cached_finish
        else:
            return False, None, None, None, None
    
    def load_generations_b(
        self, step: int, num_generations_per_model: int
    ) -> Tuple[bool, Optional[List[str]], Optional[int], Optional[List[str]], Optional[List[Optional[str]]]]:
        """
        Load Model B generations from cache.
        
        Args:
            step: Expected step number
            num_generations_per_model: Number of generations per prompt
        
        Returns:
            Tuple of (is_valid, text_responses, total_unique_prompts, cached_prompts, finish_reasons)
        """
        # Reload from disk to support multi-process usage (main rank writes; others read).
        self._load_cache()
        if "generations_b" not in self._cache_data:
            return False, None, None, None, None
        
        gen_data = self._cache_data["generations_b"]
        cache_step = gen_data.get("step")
        cache_responses = gen_data.get("text_responses", [])
        cache_total_prompts = gen_data.get("total_unique_prompts", 0)
        cached_prompts = gen_data.get("prompts", None)
        cached_finish = gen_data.get("finish_reasons", None)
        
        expected_count = cache_total_prompts * num_generations_per_model
        
        if cache_step == step and len(cache_responses) == expected_count:
            if cached_finish is not None and len(cached_finish) != expected_count:
                cached_finish = None
            return True, cache_responses, cache_total_prompts, cached_prompts, cached_finish
        else:
            return False, None, None, None, None
    
    def load_scores_a(self, step: int) -> Tuple[bool, Optional[List[float]]]:
        """
        Load Model A scores from cache.
        
        Args:
            step: Expected step number
        
        Returns:
            Tuple of (is_valid, advantages) — advantages is always a flat List[float].
        """
        self._load_cache()
        if "scores_a" not in self._cache_data:
            return False, None
        
        score_data = self._cache_data["scores_a"]
        cache_step = score_data.get("step")
        cache_advantages = score_data.get("advantages", [])
        
        if cache_step == step:
            return True, self._flatten_scores(cache_advantages)
        else:
            return False, None
    
    def load_scores_b(self, step: int) -> Tuple[bool, Optional[List[float]]]:
        """
        Load Model B scores from cache.
        
        Args:
            step: Expected step number
        
        Returns:
            Tuple of (is_valid, advantages) — advantages is always a flat List[float].
        """
        self._load_cache()
        if "scores_b" not in self._cache_data:
            return False, None
        
        score_data = self._cache_data["scores_b"]
        cache_step = score_data.get("step")
        cache_advantages = score_data.get("advantages", [])
        
        if cache_step == step:
            return True, self._flatten_scores(cache_advantages)
        else:
            return False, None
    
    def check_generations_a(self, step: int, num_generations_per_model: int) -> bool:
        """Check if Model A generations exist and are valid for given step."""
        is_valid, *_ = self.load_generations_a(step, num_generations_per_model)
        return is_valid
    
    def check_generations_b(self, step: int, num_generations_per_model: int) -> bool:
        """Check if Model B generations exist and are valid for given step."""
        is_valid, *_ = self.load_generations_b(step, num_generations_per_model)
        return is_valid
    
    def check_scores_a(self, step: int) -> bool:
        """Check if Model A scores exist and are valid for given step."""
        is_valid, _ = self.load_scores_a(step)
        return is_valid
    
    def check_scores_b(self, step: int) -> bool:
        """Check if Model B scores exist and are valid for given step."""
        is_valid, _ = self.load_scores_b(step)
        return is_valid

    def save_mean_rewards(self, step: int, mean_rewards: Dict[str, float]):
        """Append per-step mean rewards to the persistent history.

        History is stored under ``mean_rewards_history`` as a list of
        ``{"step": int, "mean_rewards": {model_id: float}}`` entries, ordered
        by step.  Duplicate steps are silently overwritten.
        """
        history: list = self._cache_data.setdefault("mean_rewards_history", [])
        # Replace existing entry for the same step, if any.
        history = [e for e in history if int(e.get("step", -1)) != int(step)]
        history.append({"step": int(step), "mean_rewards": mean_rewards})
        history.sort(key=lambda e: int(e["step"]))
        self._cache_data["mean_rewards_history"] = history
        self._save_cache()

    def save_reward_stats(self, step: int, reward_stats: Dict[str, Dict[str, float]]):
        """Append per-step reward summary stats (mean/variance) to history."""
        history: list = self._cache_data.setdefault("reward_stats_history", [])
        history = [e for e in history if int(e.get("step", -1)) != int(step)]
        history.append({"step": int(step), "reward_stats": reward_stats})
        history.sort(key=lambda e: int(e["step"]))
        self._cache_data["reward_stats_history"] = history
        self._save_cache()

    def load_mean_rewards_history(self, current_step: int, k: int) -> List[Dict[str, float]]:
        """Return the last *k* mean-reward dicts up to *current_step* (inclusive).

        Returns a list of ``{model_id: float}`` dicts, oldest first, with
        length ``min(k, available_steps)``.
        """
        history: list = self._cache_data.get("mean_rewards_history", [])
        eligible = [e for e in history if int(e["step"]) <= int(current_step)]
        eligible.sort(key=lambda e: int(e["step"]))
        return [e["mean_rewards"] for e in eligible[-k:]]

    def save_mean_sc_scores(self, step: int, mean_sc: Dict[str, float]):
        """Append per-step mean raw SCE scores (for ``sc_ratio`` capability mode)."""
        history: list = self._cache_data.setdefault("mean_sc_scores_history", [])
        history = [e for e in history if int(e.get("step", -1)) != int(step)]
        history.append({"step": int(step), "mean_sc": mean_sc})
        history.sort(key=lambda e: int(e["step"]))
        self._cache_data["mean_sc_scores_history"] = history
        self._save_cache()

    def load_mean_sc_scores_history(self, current_step: int, k: int) -> List[Dict[str, float]]:
        """Return the last *k* mean-SCE-score dicts up to *current_step* (inclusive)."""
        history: list = self._cache_data.get("mean_sc_scores_history", [])
        eligible = [e for e in history if int(e["step"]) <= int(current_step)]
        eligible.sort(key=lambda e: int(e["step"]))
        return [e["mean_sc"] for e in eligible[-k:]]

    def save_mixed_reward_dim_means(self, step: int, mv_means: Dict[str, float], int_means: Dict[str, float]):
        """Append per-step per-dimension mean rewards for mixed_reward sliding window."""
        history: list = self._cache_data.setdefault("mixed_reward_dim_means_history", [])
        history = [e for e in history if int(e.get("step", -1)) != int(step)]
        history.append({"step": int(step), "mv": mv_means, "int": int_means})
        history.sort(key=lambda e: int(e["step"]))
        self._cache_data["mixed_reward_dim_means_history"] = history
        self._save_cache()

    def load_mixed_reward_dim_means_history(
        self, current_step: int, k: int
    ) -> List[Dict[str, Dict[str, float]]]:
        """Return the last *k* per-dimension mean-reward entries up to *current_step*.

        Each entry is ``{"mv": {model_id: float}, "int": {model_id: float}}``.
        """
        history: list = self._cache_data.get("mixed_reward_dim_means_history", [])
        eligible = [e for e in history if int(e["step"]) <= int(current_step)]
        eligible.sort(key=lambda e: int(e["step"]))
        return [{"mv": e["mv"], "int": e["int"]} for e in eligible[-k:]]

    def save_capability_ratios(self, step: int, ratio_record: Dict[str, Any]):
        """Append per-step capability-ratio diagnostics to history."""
        history: list = self._cache_data.setdefault("capability_ratio_history", [])
        history = [e for e in history if int(e.get("step", -1)) != int(step)]
        entry = {"step": int(step)}
        entry.update(ratio_record)
        history.append(entry)
        history.sort(key=lambda e: int(e["step"]))
        self._cache_data["capability_ratio_history"] = history
        self._save_cache()

    def get_latest_complete_step(self, num_generations_per_model: int) -> int:
        """
        Return the latest step for which generations_a/b and scores_a/b are all present and consistent.

        Note: This cache implementation only stores the *latest* step per key (overwrites previous steps),
        so "latest complete step" is effectively the common step shared by these entries.
        Returns -1 if incomplete.
        """
        required_keys = ["generations_a", "generations_b", "scores_a", "scores_b"]
        if any(k not in self._cache_data for k in required_keys):
            return -1

        try:
            step_a = int(self._cache_data["generations_a"].get("step"))
            step_b = int(self._cache_data["generations_b"].get("step"))
            step_sa = int(self._cache_data["scores_a"].get("step"))
            step_sb = int(self._cache_data["scores_b"].get("step"))
        except Exception:
            return -1

        # All step numbers must match for the cache to be "complete" for a step.
        if not (step_a == step_b == step_sa == step_sb):
            return -1

        step = step_a
        if (
            self.check_generations_a(step, num_generations_per_model)
            and self.check_generations_b(step, num_generations_per_model)
            and self.check_scores_a(step)
            and self.check_scores_b(step)
        ):
            return step
        return -1
    
    def save_logps_a(
        self,
        step: int,
        old_per_token_logps: Optional[List[List[float]]],
        ref_per_token_logps: Optional[List[List[float]]] = None,
        *,
        alignment: str = "completion_padded_v1",
        logits_to_keep: Optional[int] = None,
        completion_lengths: Optional[List[int]] = None,
    ):
        """Save Model A logps to cache."""
        data: dict = {
            "step": step,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "alignment": str(alignment),
            "logits_to_keep": (int(logits_to_keep) if logits_to_keep is not None else None),
        }
        if completion_lengths is not None:
            data["completion_lengths"] = completion_lengths
        self._logps_cache_data["logps_a"] = data
        self._save_logps_cache()
    
    def save_logps_b(
        self,
        step: int,
        old_per_token_logps: Optional[List[List[float]]],
        ref_per_token_logps: Optional[List[List[float]]] = None,
        *,
        alignment: str = "completion_padded_v1",
        logits_to_keep: Optional[int] = None,
        completion_lengths: Optional[List[int]] = None,
    ):
        """Save Model B logps to cache."""
        data: dict = {
            "step": step,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "alignment": str(alignment),
            "logits_to_keep": (int(logits_to_keep) if logits_to_keep is not None else None),
        }
        if completion_lengths is not None:
            data["completion_lengths"] = completion_lengths
        self._logps_cache_data["logps_b"] = data
        self._save_logps_cache()

    def get_logps_alignment(self, model: str, step: int) -> Optional[str]:
        """Return alignment metadata for cached logps (always "completion_padded_v1" for new caches)."""
        self._load_logps_cache()
        key = "logps_a" if str(model).lower() in ("a", "model_a") else "logps_b"
        data = self._logps_cache_data.get(key)
        if not isinstance(data, dict):
            return None
        if data.get("step") != step:
            return None
        align = data.get("alignment")
        return str(align) if align is not None else None
    
    def load_logps_a(self, step: int) -> Tuple[bool, Optional[List[List[float]]], Optional[List[List[float]]]]:
        """
        Load Model A logps from cache.
        
        Args:
            step: Expected step number
        
        Returns:
            Tuple of (is_valid, old_per_token_logps, ref_per_token_logps)
        """
        self._load_logps_cache()
        if "logps_a" not in self._logps_cache_data:
            return False, None, None
        
        logps_data = self._logps_cache_data["logps_a"]
        cache_step = logps_data.get("step")
        
        if cache_step == step:
            old_logps = logps_data.get("old_per_token_logps")
            ref_logps = logps_data.get("ref_per_token_logps")
            return True, old_logps, ref_logps
        else:
            return False, None, None
    
    def load_logps_b(self, step: int) -> Tuple[bool, Optional[List[List[float]]], Optional[List[List[float]]]]:
        """
        Load Model B logps from cache.
        
        Args:
            step: Expected step number
        
        Returns:
            Tuple of (is_valid, old_per_token_logps, ref_per_token_logps)
        """
        self._load_logps_cache()
        if "logps_b" not in self._logps_cache_data:
            return False, None, None
        
        logps_data = self._logps_cache_data["logps_b"]
        cache_step = logps_data.get("step")
        
        if cache_step == step:
            old_logps = logps_data.get("old_per_token_logps")
            ref_logps = logps_data.get("ref_per_token_logps")
            return True, old_logps, ref_logps
        else:
            return False, None, None
    
    def load_completion_lengths(self, model: str, step: int) -> Optional[List[int]]:
        """Load per-sample completion lengths cached alongside logps.

        Returns None if the cache entry is missing or was written before
        completion_lengths were stored (backward compatible).
        """
        self._load_logps_cache()
        key = "logps_a" if str(model).lower() in ("a", "model_a") else "logps_b"
        data = self._logps_cache_data.get(key)
        if not isinstance(data, dict):
            return None
        if data.get("step") != step:
            return None
        cl = data.get("completion_lengths")
        if cl is not None and isinstance(cl, list):
            return cl
        return None

    def check_logps_a(self, step: int) -> bool:
        """Check if Model A logps exist and are valid for given step."""
        is_valid, _, _ = self.load_logps_a(step)
        return is_valid
    
    def check_logps_b(self, step: int) -> bool:
        """Check if Model B logps exist and are valid for given step."""
        is_valid, _, _ = self.load_logps_b(step)
        return is_valid

    # ------------------------------------------------------------------
    # Cross old logps (retokenize pipeline):
    # cross_logps_a = Model A's old logps on Model B's responses
    # cross_logps_b = Model B's old logps on Model A's responses
    # ------------------------------------------------------------------

    def save_cross_logps(
        self,
        model: str,
        step: int,
        old_per_token_logps: Optional[List[List[float]]],
        *,
        alignment: str = "completion_padded_v1",
        logits_to_keep: Optional[int] = None,
        completion_lengths: Optional[List[int]] = None,
    ):
        """Save cross old logps (model's logps on the *other* model's responses)."""
        key = "cross_logps_a" if str(model).lower() in ("a", "model_a") else "cross_logps_b"
        data: dict = {
            "step": step,
            "old_per_token_logps": old_per_token_logps,
            "alignment": str(alignment),
            "logits_to_keep": (int(logits_to_keep) if logits_to_keep is not None else None),
        }
        if completion_lengths is not None:
            data["completion_lengths"] = completion_lengths
        self._logps_cache_data[key] = data
        self._save_logps_cache()

    def load_cross_logps(
        self, model: str, step: int,
    ) -> Tuple[bool, Optional[List[List[float]]]]:
        """Load cross old logps for *model* (its logps on the other model's responses).

        Returns:
            Tuple of (is_valid, old_per_token_logps).
        """
        self._load_logps_cache()
        key = "cross_logps_a" if str(model).lower() in ("a", "model_a") else "cross_logps_b"
        data = self._logps_cache_data.get(key)
        if not isinstance(data, dict):
            return False, None
        if data.get("step") != step:
            return False, None
        return True, data.get("old_per_token_logps")

    def check_cross_logps(self, model: str, step: int) -> bool:
        """Check if cross old logps exist and are valid for given model/step."""
        is_valid, _ = self.load_cross_logps(model, step)
        return is_valid

    def load_cross_completion_lengths(self, model: str, step: int) -> Optional[List[int]]:
        """Load completion lengths from cross old logps cache entry."""
        self._load_logps_cache()
        key = "cross_logps_a" if str(model).lower() in ("a", "model_a") else "cross_logps_b"
        data = self._logps_cache_data.get(key)
        if not isinstance(data, dict):
            return None
        if data.get("step") != step:
            return None
        cl = data.get("completion_lengths")
        if cl is not None and isinstance(cl, list):
            return cl
        return None

    def save_checkpoint_info(
        self,
        step: int,
        start_index: int,
        sampled_indices: List[int],
        *,
        sampling_strategy: Optional[str] = None,
        epoch: Optional[int] = None,
        pos: Optional[int] = None,
        epoch_len: Optional[int] = None,
        num_problems: Optional[int] = None,
        problems_per_step: Optional[int] = None,
        seed: Optional[int] = None,
        shuffle_dataset: Optional[bool] = None,
        perm_seed: Optional[int] = None,
    ):
        """Save checkpoint info (sampling state) to cache. Only keeps the latest step."""
        # Overwrite checkpoint_info with only the latest step (don't accumulate)
        rec: Dict[str, Any] = {
            "step": step,
            "start_index": start_index,
            "sampled_indices": sampled_indices,
        }
        # Optional extended sampling state (backward compatible with old readers).
        if sampling_strategy is not None:
            rec["sampling_strategy"] = str(sampling_strategy)
        if epoch is not None:
            rec["epoch"] = int(epoch)
        if pos is not None:
            rec["pos"] = int(pos)
        if epoch_len is not None:
            rec["epoch_len"] = int(epoch_len)
        if num_problems is not None:
            rec["num_problems"] = int(num_problems)
        if problems_per_step is not None:
            rec["problems_per_step"] = int(problems_per_step)
        if seed is not None:
            rec["seed"] = int(seed)
        if shuffle_dataset is not None:
            rec["shuffle_dataset"] = bool(shuffle_dataset)
        if perm_seed is not None:
            rec["perm_seed"] = int(perm_seed)

        self._cache_data["checkpoint_info"] = rec
        self._save_cache()

    def load_checkpoint_info(self, step: int) -> Optional[Dict[str, Any]]:
        """Return a defensive copy of the sampling metadata for *step*."""
        self._load_cache()
        rec = self._cache_data.get("checkpoint_info")
        if not isinstance(rec, dict):
            return None
        try:
            if int(rec.get("step")) != int(step):
                return None
        except (TypeError, ValueError):
            return None
        # Round-trip through JSON-compatible containers so callers cannot mutate
        # the cache manager's in-memory state through nested lists/dicts.
        return json.loads(json.dumps(rec))

    def save_metadata(self, key: str, value: Any):
        """Store arbitrary metadata in the cache (e.g. model identity for staleness detection)."""
        if "metadata" not in self._cache_data:
            self._cache_data["metadata"] = {}
        self._cache_data["metadata"][str(key)] = value
        self._save_cache()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata stored via save_metadata."""
        self._load_cache()
        meta = self._cache_data.get("metadata")
        if not isinstance(meta, dict):
            return default
        return meta.get(str(key), default)
