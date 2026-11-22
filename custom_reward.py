import gymnasium as gym
from stable_baselines3 import PPO

# Define a Custom Wrapper to shape the rewards
class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        # 1. Get the standard results from the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Apply Reward Shaping: Add a tiny time penalty (-0.05 per frame) 
        # so the agent learns to land efficiently instead of wasting time.
        reward -= 0.05
        
        return observation, reward, terminated, truncated, info

# 1. Create the base environment and wrap it
base_env = gym.make("LunarLander-v3")
env = TimePenaltyWrapper(base_env)

# 2. Initialize PPO with our shaped environment
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)

print("--- Training with Custom Reward Shaping (50,000 steps) ---")
model.learn(total_timesteps=50000)

# 3. Save the shaped model
model.save("ppo_lunar_lander_shaped")
print("Shaped model saved successfully as 'ppo_lunar_lander_shaped.zip'!")

env.close()