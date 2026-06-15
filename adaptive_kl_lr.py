from typing import Callable, Optional
from stable_baselines3.common.callbacks import BaseCallback


class AdaptiveKLLRCallback(BaseCallback):
    """KL-adaptive LR (Rudin et al., "Learning to Walk in Minutes...") for PPO,
    composed on top of a base linear-decay schedule:  effective LR = base * mult.

    Per rollout (after train() logged approx_kl):
      - PPO early-stopped (n_updates_done < n_epochs): LR /= early_stop_decay
      - KL > 2*target_kl:                              LR /= 1.5   (strong decay)
      - KL < 0.5*target_kl:                            LR *= 1.5   (speed up, capped)
      - else: hold

    BLOWUP FIX (`max_multiplier`, default 1.0):
      The original controller's flaw: near-optimal the gradient is tiny, so KL
      stays < 0.5*target no matter the LR, and the `*1.5` branch ratchets LR up to
      `lr_cap_late` (8e-4, 8x base). LR sits dangerously high until a gradient spike
      → catastrophic update → collapse (observed: the best run died at 9.5M with
      kl_lr_multiplier→1.57). Clamping the multiplier to <= max_multiplier means the
      effective LR can never EXCEED the planned base schedule — the controller may
      only CUT LR for safety, never inflate it. Set max_multiplier>1.0 to restore
      the (unstable) original upward behaviour.
    """

    def __init__(
        self,
        target_kl: float,
        lr_floor: float = 1e-5,
        lr_cap_early: float = 1e-2,
        lr_cap_late: float = 8e-4,
        timestep_threshold: int = 2_000_000,
        early_stop_decay: float = 1.2,
        max_multiplier: float = 1.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_kl = target_kl
        self.lr_floor = lr_floor
        self.lr_cap_early = lr_cap_early
        self.lr_cap_late = lr_cap_late
        self.timestep_threshold = timestep_threshold
        self.early_stop_decay = early_stop_decay
        self.max_multiplier = max_multiplier
        self.multiplier = [1.0]
        self._original_schedule: Optional[Callable[[float], float]] = None
        self._n_updates_before: int = 0
        self._n_epochs: int = 1

    def _on_training_start(self) -> None:
        self._original_schedule = self.model.lr_schedule
        original = self._original_schedule
        multiplier_ref = self.multiplier

        def adaptive_schedule(progress_remaining: float) -> float:
            return original(progress_remaining) * multiplier_ref[0]

        self.model.lr_schedule = adaptive_schedule
        self._n_epochs = getattr(self.model, "n_epochs", 1)

    def _on_rollout_end(self) -> None:
        self._n_updates_before = self.model._n_updates

    def _on_rollout_start(self) -> bool:
        kl = self.model.logger.name_to_value.get("train/approx_kl", None)
        if kl is None:
            return True

        n_updates_done = self.model._n_updates - self._n_updates_before
        early_stopped = n_updates_done < self._n_epochs
        current_lr = self.model.policy.optimizer.param_groups[0]["lr"]

        if early_stopped:
            new_lr = max(self.lr_floor, current_lr / self.early_stop_decay)
            reason = f"early_stop({n_updates_done}/{self._n_epochs})"
        elif kl > 2.0 * self.target_kl:
            new_lr = max(self.lr_floor, current_lr / 1.5)
            reason = "kl>2*target"
        elif kl < 0.5 * self.target_kl:
            cap = self.lr_cap_late if self.num_timesteps > self.timestep_threshold else self.lr_cap_early
            new_lr = min(cap, current_lr * 1.5)
            reason = "kl<0.5*target"
        else:
            return True

        progress_remaining = 1.0 - self.num_timesteps / self.model._total_timesteps
        if self._original_schedule is not None:
            base_lr = self._original_schedule(progress_remaining)
            if base_lr > 0:
                # FIX: never let the effective LR exceed the base schedule.
                self.multiplier[0] = min(self.max_multiplier, new_lr / base_lr)

        self.logger.record("train/kl_lr_multiplier", self.multiplier[0])
        self.logger.record("train/adaptive_lr", self._original_schedule(progress_remaining) * self.multiplier[0])
        self.logger.record("train/approx_kl_for_lr_adj", kl)
        self.logger.record("train/early_stopped", int(early_stopped))
        if self.verbose >= 1:
            print(f"[AdaptiveKLLR] step={self.num_timesteps} kl={kl:.4f} reason={reason} "
                  f"mult={self.multiplier[0]:.4f}")
        return True

    def _on_step(self) -> bool:
        return True
