from typing import Callable, Optional
from stable_baselines3.common.callbacks import BaseCallback


class AdaptiveKLLRCallback(BaseCallback):
    """
    Adjusts learning rate multiplicatively based on observed KL divergence,
    while remaining compatible with an existing linear LR decay schedule.

    The effective LR = base_linear_schedule(progress_remaining) * multiplier.

    Detection of PPO early stopping: SB3 increments model._n_updates once per
    completed epoch inside the epoch loop. We snapshot it in _on_rollout_end
    (just before train()) and compare in _on_rollout_start (just after). If
    fewer than n_epochs completed, early stopping fired.

    Three adjustment zones (checked each rollout):
      - Early stopped (n_updates_done < n_epochs): mild decay  (LR / early_stop_decay)
      - KL > 2 * target_kl:                        strong decay (LR / 1.5)
      - KL < 0.5 * target_kl:                      increase     (LR * 1.5, capped)

    Args:
        target_kl:            Target KL divergence (same value passed to PPO).
        lr_floor:             Minimum allowed effective learning rate.
        lr_cap_early:         Maximum allowed LR before timestep_threshold.
        lr_cap_late:          Maximum allowed LR after timestep_threshold.
        timestep_threshold:   Timestep at which the cap switches from early to late.
        early_stop_decay:     Divisor applied when PPO early stopping is detected.
        verbose:              Verbosity level.
    """

    def __init__(
        self,
        target_kl: float,
        lr_floor: float = 1e-5,
        lr_cap_early: float = 1e-2,
        lr_cap_late: float = 8e-4,
        timestep_threshold: int = 2_000_000,
        early_stop_decay: float = 1.2,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_kl = target_kl
        self.lr_floor = lr_floor
        self.lr_cap_early = lr_cap_early
        self.lr_cap_late = lr_cap_late
        self.timestep_threshold = timestep_threshold
        self.early_stop_decay = early_stop_decay
        self.multiplier = 1.0
        self._original_schedule: Optional[Callable[[float], float]] = None
        self._n_updates_before: int = 0
        self._n_epochs: int = 1  # set in _on_training_start from model

    def _on_training_start(self) -> None:
        # Save the original (linear decay) schedule and wrap it with the multiplier.
        self._original_schedule = self.model.lr_schedule
        original = self._original_schedule
        callback = self

        def adaptive_schedule(progress_remaining: float) -> float:
            return original(progress_remaining) * callback.multiplier

        self.model.lr_schedule = adaptive_schedule
        self._n_epochs = getattr(self.model, "n_epochs", 1)

    def _on_rollout_end(self) -> None:
        # Snapshot _n_updates just before train() is called.
        self._n_updates_before = self.model._n_updates

    def _on_rollout_start(self) -> bool:
        kl = self.model.logger.name_to_value.get("train/approx_kl", None)
        if kl is None:
            return True  # first iteration — no KL yet

        # Detect early stopping: SB3 increments _n_updates once per completed
        # epoch, so fewer increments than n_epochs means the loop was cut short.
        n_updates_done = self.model._n_updates - self._n_updates_before
        early_stopped = n_updates_done < self._n_epochs

        current_lr = self.model.policy.optimizer.param_groups[0]["lr"]

        if early_stopped:
            new_lr = max(self.lr_floor, current_lr / self.early_stop_decay)
            reason = f"early_stop({n_updates_done}/{self._n_epochs} epochs)"
        elif kl > 2.0 * self.target_kl:
            new_lr = max(self.lr_floor, current_lr / 1.5)
            reason = "kl>2*target"
        elif kl < 0.5 * self.target_kl:
            cap = self.lr_cap_late if self.num_timesteps > self.timestep_threshold else self.lr_cap_early
            new_lr = min(cap, current_lr * 1.5)
            reason = "kl<0.5*target"
        else:
            return True  # KL within acceptable band

        # Recompute multiplier so adaptive_schedule returns new_lr at current progress.
        progress_remaining = 1.0 - self.num_timesteps / self.model._total_timesteps
        if self._original_schedule is not None:
            base_lr = self._original_schedule(progress_remaining)
            if base_lr > 0:
                self.multiplier = new_lr / base_lr

        self.logger.record("train/kl_lr_multiplier", self.multiplier)
        self.logger.record("train/adaptive_lr", new_lr)
        self.logger.record("train/approx_kl_for_lr_adj", kl)
        self.logger.record("train/early_stopped", int(early_stopped))

        if self.verbose >= 1:
            print(
                f"[AdaptiveKLLR] step={self.num_timesteps} kl={kl:.4f} "
                f"reason={reason} new_lr={new_lr:.2e} multiplier={self.multiplier:.4f}"
            )

        return True

    def _on_step(self) -> bool:
        return True
