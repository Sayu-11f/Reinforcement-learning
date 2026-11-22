import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment (No render during training so it goes super fast)
env = gym.make("LunarLander-v3")

# 2. Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)

print("--- Starting Training (50,000 steps) ---")
# Train for 50,000 steps (takes about 1-2 minutes on a modern CPU)
model.learn(total_timesteps=50000)

# 3. Save the trained model weights
model.save("ppo_lunar_lander")
print("Model saved successfully as 'ppo_lunar_lander.zip'!")

# 4. Close training environment and open a rendering environment to WATCH it play!
env.close()
eval_env = gym.make("LunarLander-v3", render_mode="human")

print("--- Watching the trained agent play ---")
obs, info = eval_env.reset()
for _ in range(1000):  # Run for 1000 frames (~20 seconds)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()