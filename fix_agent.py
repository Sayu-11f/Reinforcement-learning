import gymnasium as gym
from stable_baselines3 import PPO

# Define a Fixed Wrapper: Rewards progress and penalizes hovering/wasting time
class FixedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset distance tracking on new episode
        # obs[0] is horizontal position, obs[1] is vertical height above pad
        self.prev_distance = (obs[0]**2 + obs[1]**2)**0.5
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate current distance to the landing pad target (0,0)
        current_distance = (observation[0]**2 + observation[1]**2)**0.5
        
        # FIX 1: Reward the agent for getting CLOSER (change in distance between frames)
        distance_progress = self.prev_distance - current_distance
        reward += distance_progress * 10.0
        self.prev_distance = current_distance
        
        # FIX 2: Add a strict time penalty per frame so hovering continuously drains points
        reward -= 0.1
        
        return observation, reward, terminated, truncated, info

# 1. Create environment with the fixed wrapper
base_env = gym.make("LunarLander-v3")
env = FixedRewardWrapper(base_env)

# 2. Train PPO for a solid session
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)

print("--- Training with Fixed Reward Shaping (500,000 steps) ---")
model.learn(total_timesteps=500000)

# 3. Save the fixed model
model.save("ppo_lunar_lander_fixed")
print("Fixed model saved successfully as 'ppo_lunar_lander_fixed.zip'!")

env.close()