from typing import Callable, Optional
from stable_baselines3.common.callbacks import BaseCallback


class AdaptiveKLLRCallback(BaseCallback):
    """
    Adjusts learning rate multiplicatively based on observed KL divergence,
    while remaining compatible with an existing linear LR decay schedule.

    The effective LR = base_linear_schedule(progress_remaining) * multiplier.

    On each rollout start (after the previous train() step has logged approx_kl):
      - If KL > 2 * target_kl  → decrease multiplier (LR / 1.5), floor at lr_floor
      - If KL < 0.5 * target_kl → increase multiplier (LR * 1.5), capped at lr_cap_early
        or lr_cap_late (after timestep_threshold steps)

    Args:
        target_kl:          Target KL divergence (same value passed to PPO).
        lr_floor:           Minimum allowed effective learning rate.
        lr_cap_early:       Maximum allowed LR before timestep_threshold.
        lr_cap_late:        Maximum allowed LR after timestep_threshold.
        timestep_threshold: Timestep at which the cap switches from early to late.
        verbose:            Verbosity level.
    """

    def __init__(
        self,
        target_kl: float,
        lr_floor: float = 1e-5,
        lr_cap_early: float = 1e-2,
        lr_cap_late: float = 8e-4,
        timestep_threshold: int = 2_000_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_kl = target_kl
        self.lr_floor = lr_floor
        self.lr_cap_early = lr_cap_early
        self.lr_cap_late = lr_cap_late
        self.timestep_threshold = timestep_threshold
        self.multiplier = 1.0
        self._original_schedule: Optional[Callable[[float], float]] = None

    def _on_training_start(self) -> None:
        # Save the original (linear decay) schedule and wrap it with the multiplier.
        self._original_schedule = self.model.lr_schedule
        original = self._original_schedule
        callback = self

        def adaptive_schedule(progress_remaining: float) -> float:
            return original(progress_remaining) * callback.multiplier

        self.model.lr_schedule = adaptive_schedule

    def _on_rollout_start(self) -> bool:
        # approx_kl is logged by PPO's train() from the *previous* iteration.
        # name_to_value holds the last recorded value for each key.
        kl = self.model.logger.name_to_value.get("train/approx_kl", None)
        if kl is None:
            return True  # first iteration — no KL yet

        # Current effective LR as set in the optimizer by _update_learning_rate().
        current_lr = self.model.policy.optimizer.param_groups[0]["lr"]

        if kl > 2.0 * self.target_kl:
            new_lr = max(self.lr_floor, current_lr / 1.5)
        elif kl < 0.5 * self.target_kl:
            cap = self.lr_cap_late if self.num_timesteps > self.timestep_threshold else self.lr_cap_early
            new_lr = min(cap, current_lr * 1.5)
        else:
            # KL is within the acceptable band — no adjustment needed.
            return True

        # Recompute multiplier so the adaptive schedule produces new_lr at the
        # current progress. SB3 will call lr_schedule(progress_remaining) in
        # _update_learning_rate() at the start of the next train() step.
        progress_remaining = 1.0 - self.num_timesteps / self.model._total_timesteps
        if self._original_schedule is not None:
            base_lr = self._original_schedule(progress_remaining)
            if base_lr > 0:
                self.multiplier = new_lr / base_lr

        self.logger.record("train/kl_lr_multiplier", self.multiplier)
        self.logger.record("train/adaptive_lr", new_lr)
        self.logger.record("train/approx_kl_for_lr_adj", kl)

        if self.verbose >= 1:
            print(
                f"[AdaptiveKLLR] step={self.num_timesteps} kl={kl:.4f} "
                f"target={self.target_kl} new_lr={new_lr:.2e} multiplier={self.multiplier:.4f}"
            )

        return True

    def _on_step(self) -> bool:
        return True
