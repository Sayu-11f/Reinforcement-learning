# Evaluating Reinforcement Learning Robustness Under Environmental Uncertainties

This project trains PPO (Proximal Policy Optimization) agents on the `LunarLander-v3` environment using reward shaping and curriculum learning to improve robustness against dynamic environmental uncertainties such as sensor noise, engine failures, wind, and observation discretization.

It compares multiple model variants (baseline, custom reward, curriculum v1, curriculum v2, reward-hacked) across six disturbance conditions and generates visualization graphs.

## Project Structure

```
rl_project/
├── README.md
├── compare_models.py                 # Evaluates all models across test conditions & generates graphs
├── custom_reward.py                  # Trains custom reward shaping model (50k steps)
├── discretization_wrapper.py         # Observation discretization wrapper utility
├── fix_agent.py                      # Trains baseline fixed-reward model (500k steps)
├── robust_lunar_lander_curriculum.py # Curriculum training v1 (3 stages, 100k steps each)
├── robust_lunar_lander_curriculum2.py# Curriculum training v2 (3 stages, 500k steps each)
├── ppo_lunar_lander_fixed.zip        # Trained baseline fixed-reward model
├── robust_lunar_lander_curriculum.zip# Trained curriculum v1 model
├── robust_lunar_lander_curriculum2.zip# Trained curriculum v2 model
├── desktop.ini                       # Windows system file (not project-related)
├── Evaluating Reinforcement Learning Robustness Under Dynamic Environmental Uncertainties_ A Curriculum Fine-Tuning Approach (1).pdf  # Reference paper
├── __pycache__/                      # Python bytecode cache (auto-generated)
├── rl_env/                           # Virtual environment
├── Graphs/                           # Generated comparison graphs (PNG/JPG)
├── logs/                             # Training monitor CSVs (stage_1, stage_2, stage_3)
│   ├── monitor.csv
│   ├── stage_1/monitor.csv
│   ├── stage_2/monitor.csv
│   └── stage_3/monitor.csv
├── Other_models/                     # Additional model variants
│   ├── ppo_lunar_lander.zip
│   ├── ppo_lunar_lander_shaped.zip
│   ├── ppo_lunar_lander_broken.zip
│   └── custom_lunar_lander_model.zip
└── Other_scripts/                    # Helper / experimental scripts
    ├── Break_agent.py                # Trains reward-hacked "broken" agent
    ├── train_and_watch.py            # Basic train + render script (50k steps)
    └── watch_model.py                # Visualizes a trained model in human render mode
```

## Features

- **Curriculum Learning**: Progressive training across 3 stages — Baseline → Moderate Uncertainty → Full Chaos
- **Dynamic Uncertainty Wrapper**: Injects sensor noise, engine failure probability, observation discretization, and wind disturbances
- **Reward Shaping**: Custom reward functions to encourage efficient landing and penalize hovering
- **Model Comparison**: Evaluates Naive Agent, Curriculum v1, and Curriculum v2 across 6 test conditions:
  1. Level 0 (Quiet) — no disturbances
  2. Level 1 (Noise) — sensor noise
  3. Level 2 (Failures) — engine failure probability
  4. Level 3 (Wind) — wind + turbulence
  5. Level 4 (Quantization) — observation discretization
  6. Level 5 (Chaos) — combined disturbances
- **Visualization**: Success rate trends, performance over time with variance shading, reward box plots, and stage training curves

## Requirements

Install dependencies in your virtual environment:

```bash
pip install gymnasium stable-baselines3 matplotlib seaborn pandas numpy torch
```

Or activate the existing environment:

```bash
.\rl_env\Scripts\Activate.ps1
```

## How to Train Models

### 1. Baseline Fixed-Reward Model
```bash
python fix_agent.py
```
Trains for 500,000 steps with distance-progress rewards and time penalty. Saves as `ppo_lunar_lander_fixed.zip`.

### 2. Custom Reward Shaping Model
```bash
python custom_reward.py
```
Trains for 50,000 steps with a small per-frame time penalty. Saves as `ppo_lunar_lander_shaped.zip`.

### 3. Curriculum Learning Model v1
```bash
python robust_lunar_lander_curriculum.py
```
Three stages of 100,000 steps each. Loads `ppo_lunar_lander_fixed.zip` as pretrained weights. Saves as `robust_lunar_lander_curriculum.zip`.

### 4. Curriculum Learning Model v2
```bash
python robust_lunar_lander_curriculum2.py
```
Same curriculum as v1 but with 500,000 steps per stage. Saves as `robust_lunar_lander_curriculum2.zip`.

### 5. Reward-Hacked "Broken" Agent (Experimental)
```bash
python Other_scripts/Break_agent.py
```
Trains for 50,000 steps with a faulty reward rule that rewards the agent for staying upright and slow. Saves as `ppo_lunar_lander_broken.zip`.

## How to Evaluate & Compare Models

Run the full evaluation suite:

```bash
python compare_models.py
```

This loads all three main models (`ppo_lunar_lander_fixed.zip`, `robust_lunar_lander_curriculum.zip`, `robust_lunar_lander_curriculum2.zip`) and evaluates them across all 6 test conditions with 100 episodes each. It generates:

- `success_rate_trend_graph.png` — Success rate vs disturbance level
- `performance_over_time_shaded.png` — Performance with variance shading
- `reward_boxplot_with_strips.png` — Reward distribution box + strip plot

Individual stage training curves are saved during curriculum training:
- `curriculum_stage_1_curve.png`
- `curriculum_stage_2_curve.png`
- `curriculum_stage_3_curve.png`

## How to Watch a Model Play

```bash
python Other_scripts/watch_model.py
```
Loads `ppo_lunar_lander_fixed.zip` and runs 5 episodes with human rendering.

Or use the combined train-and-watch script:

```bash
python Other_scripts/train_and_watch.py
```
Trains a fresh model for 50,000 steps, then renders it playing.

## Key Scripts Summary

| Script | Steps | Purpose |
|--------|-------|---------|
| `fix_agent.py` | 500k | Baseline fixed-reward training |
| `custom_reward.py` | 50k | Custom reward shaping training |
| `robust_lunar_lander_curriculum.py` | 3×100k | Curriculum training v1 |
| `robust_lunar_lander_curriculum2.py` | 3×500k | Curriculum training v2 |
| `compare_models.py` | — | Cross-model evaluation & graphing |
| `Other_scripts/Break_agent.py` | 50k | Reward-hacked agent (ablation) |
| `Other_scripts/train_and_watch.py` | 50k | Quick train + render demo |
| `Other_scripts/watch_model.py` | — | Model visualization |

## Conclusion
**Research Findings**

The experiments demonstrate that curriculum fine-tuning using pretrained PPO models can improve robustness under simulated environmental disturbances. By building upon an existing model with established flight-control capabilities, subsequent training can focus on adapting to increasingly complex conditions.

Across the six evaluation environments, **Curriculum V1 outperformed the baseline agent in all six tests**, demonstrating the most consistent overall performance. Curriculum V2 achieved the highest success rate in the Full Adaptive Chaos environment (6%), compared with 2% for Curriculum V1 and 0% for the baseline.

An important finding was that increased training duration did not necessarily produce better overall performance. Despite receiving approximately 1.5 million training steps, Curriculum V2 was outperformed by Curriculum V1 in four of the six environments, despite Curriculum V1 receiving only 300,000 steps. However, Curriculum V2 achieved stronger performance in the most challenging environment and recorded the highest evaluation reward among the three models.

These results suggest that progressively challenging training environments and pretrained models are promising approaches to improving RL robustness. However, the unequal training budgets and use of single training seeds limit the strength of these conclusions. Further experiments with controlled training budgets and multiple random seeds are needed to evaluate the generalisability of the findings.
