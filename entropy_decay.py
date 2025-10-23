import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EntropyDecayCallback(BaseCallback):
    # ... (Initialization code remains the same) ...
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, total_timesteps: int, verbose: int = 0):
        super(EntropyDecayCallback, self).__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps
    
    def _on_training_start(self) -> None:
        # Ensure the initial value is set correctly at the start
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = self.initial_ent_coef
        elif self.verbose > 0:
            print("Warning: Model does not have a modifiable 'ent_coef' attribute.")
            
    def _on_step(self) -> bool:
        if not hasattr(self.model, 'ent_coef'):
            # Stop if the model doesn't support a decaying entropy coefficient
            return True

        # 1. Calculate and update the new entropy coefficient
        progress = np.clip(self.num_timesteps / self.total_timesteps, 0.0, 1.0) 
        new_ent_coef = self.initial_ent_coef * (1.0 - progress) + self.final_ent_coef * progress
        self.model.ent_coef = new_ent_coef
        
        # 2. LOG THE NEW ENTROPY COEFFICIENT
        # This will create a 'train/entropy_coefficient' curve in TensorBoard
        self.logger.record('train/entropy_coefficient', new_ent_coef)
        
        if self.verbose > 1:
            print(f"Timesteps: {self.num_timesteps}, New ent_coef: {new_ent_coef:.6f}")

        return True