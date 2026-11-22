import gymnasium as gym
from stable_baselines3 import PPO

# Define a Wrapper with a faulty reward rule (Reward Hacking)
class BreakTheAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Unpack observation array to check tilt and speed
        # obs[4] is tilt (radians), obs[3] is up/down speed, obs[2] is sideways speed
        tilt = abs(observation[4])
        up_down_speed = abs(observation[3])
        side_speed = abs(observation[2])
        
        # FAULTY RULE: Give +1.0 bonus every frame the rocket is relatively upright and slow
        if tilt < 0.1 and up_down_speed < 0.2 and side_speed < 0.2:
            reward += 1.0  
            
        return observation, reward, terminated, truncated, info

# 1. Create the environment with our broken reward rule
base_env = gym.make("LunarLander-v3")
env = BreakTheAgentWrapper(base_env)

# 2. Train PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)

print("--- Training with Faulty Reward Hacking (50000 steps) ---")
model.learn(total_timesteps=50000)

# 3. Save the broken model
model.save("ppo_lunar_lander_broken")
print("Broken model saved as 'ppo_lunar_lander_broken.zip'!")

env.close()