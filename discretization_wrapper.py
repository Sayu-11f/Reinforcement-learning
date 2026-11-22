import gymnasium as gym
import numpy as np

class DiscretizationWrapper(gym.ObservationWrapper):
    """
    Wraps the environment to bin continuous observation states 
    into discrete quantization steps.
    """
    def __init__(self, env, precision_factor=10.0):
        super().__init__(env)
        self.precision_factor = precision_factor  # Higher = finer grid, Lower = coarser bins

    def observation(self, obs):
        # Quantize continuous states (e.g., coordinates, velocities, angles)
        # This rounds values to discrete steps defined by the precision factor
        return np.round(obs * self.precision_factor) / self.precision_factor