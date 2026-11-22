import gymnasium as gym
from stable_baselines3 import PPO

# 1. Load your trained model
model = PPO.load("ppo_lunar_lander_fixed")

# 2. Create the environment with rendering enabled
env = gym.make("LunarLander-v3", render_mode="human")

# 3. Run 5 test episodes so you can watch it multiple times
for episode in range(1, 6):
    obs, info = env.reset()
    total_reward = 0
    done = False
    
    print(f"--- Starting Episode {episode} ---")
    
    while not done:
        # Use deterministic=True so the agent makes its best decision every time
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        done = terminated or truncated

    print(f"Episode {episode} finished with total reward: {total_reward:.2f}")

env.close()