import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

class LunarLanderUncertaintyWrapper(gym.Wrapper):
    def __init__(self, env, noise_scale=0.0, failure_prob=0.0):
        super().__init__(env)
        self.noise_scale = noise_scale
        self.failure_prob = failure_prob

    def step(self, action):
        if self.failure_prob > 0 and np.random.rand() < self.failure_prob:
            action = 0  # Engine failure: force do-nothing

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Soften extreme negative penalties slightly to prevent "crash loops"
        if reward < -50:
            reward = reward * 0.5 

        if self.noise_scale > 0:
            obs = obs + np.random.normal(0, self.noise_scale, size=obs.shape)

        return obs, reward, terminated, truncated, info

def plot_training_progress(log_dir, stage_title, filename):
    """Plots and saves the episode reward curve for a specific stage."""
    df = load_results(log_dir)
    x, y = ts2xy(df, "timesteps")
    
    def smooth(values, window=10):
        if len(values) < window:
            return values
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')

    smoothed_y = smooth(y, window=15)
    x_smoothed = x[:len(smoothed_y)]

    plt.figure(figsize=(10, 5), facecolor='#1e1e1e')
    ax = plt.axes()
    ax.set_facecolor('#121212')

    plt.plot(x, y, color='cyan', alpha=0.25, label='Raw Episode Reward')
    plt.plot(x_smoothed, smoothed_y, color='cyan', linewidth=2, label='Smoothed Trend')

    plt.title(f"Training Progress: {stage_title}", color='white', fontsize=14)
    plt.xlabel("Total Timesteps (Stage)", color='white', fontsize=12)
    plt.ylabel("Episode Reward", color='white', fontsize=12)
    plt.tick_params(colors='white')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.legend(facecolor='#1e1e1e', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='#1e1e1e')
    plt.show()

def train_curriculum_model():
    stages = [
        {"name": "Stage 1: Fine-Tuning Baseline", "steps": 100000, "wind": False, "wind_p": 0.0, "turb": 0.0, "noise": 0.00, "fail": 0.0, "log_dir": "./logs/stage_1/"},
        {"name": "Stage 2: Moderate Uncertainty", "steps": 100000, "wind": True,  "wind_p": 5.0, "turb": 0.8, "noise": 0.02, "fail": 0.1, "log_dir": "./logs/stage_2/"},
        {"name": "Stage 3: Full Chaos", "steps": 100000, "wind": True,  "wind_p": 10.0,"turb": 1.5, "noise": 0.05, "fail": 0.2, "log_dir": "./logs/stage_3/"}
    ]
    
    pretrained_path = "ppo_lunar_lander_fixed.zip"
    model = None
    
    for i, stage in enumerate(stages):
        print(f"\n--- Starting {stage['name']} ({stage['steps']} steps) ---")
        os.makedirs(stage["log_dir"], exist_ok=True)
        
        base_env = gym.make(
            "LunarLander-v3",
            enable_wind=stage["wind"],
            wind_power=stage["wind_p"],
            turbulence_power=stage["turb"]
        )
        
        env = LunarLanderUncertaintyWrapper(base_env, noise_scale=stage["noise"], failure_prob=stage["fail"])
        env = Monitor(env, stage["log_dir"])
        
        if model is None:
            if os.path.exists(pretrained_path):
                print(f"Loading pre-trained weights from {pretrained_path}...")
                model = PPO.load(pretrained_path, env=env, verbose=1)
            else:
                print("No pre-trained model found, initializing from scratch...")
                model = PPO("MlpPolicy", env, verbose=1)
        else:
            model.set_env(env)
            
        model.learn(total_timesteps=stage["steps"])
        env.close()
        
        plot_filename = f"curriculum_stage_{i+1}_curve.png"
        plot_training_progress(stage["log_dir"], stage["name"], plot_filename)
        
    model.save("robust_lunar_lander_curriculum")
    print("\n--- Curriculum Training Complete! Model saved as 'robust_lunar_lander_curriculum' ---")

if __name__ == "__main__":
    train_curriculum_model()