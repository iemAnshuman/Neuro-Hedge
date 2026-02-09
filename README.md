# Neuro-Hedge: Guided DRL for Derivatives Hedging

**Neuro-Hedge** is a Deep Reinforcement Learning (DRL) framework designed to optimize option hedging strategies in realistic markets with transaction costs. Unlike traditional Black-Scholes-Merton (BSM) models that assume frictionless trading, Neuro-Hedge uses a **Guided DDPG** approach to balance the trade-off between minimizing hedging error and reducing execution costs.

## 🚀 Key Features

* **Guided DDPG Architecture:** Overcomes the "cold-start" problem in financial RL by using the analytical BSM solution to guide exploration.
* **Teacher Forcing Curriculum:** Implements probabilistic teacher forcing that decays exponentially, stabilizing early training.
* **Cost-Aware Optimization:** Explicitly accounts for linear transaction costs (spreads/commissions) in the reward function, learning to "smooth" trading activity.
* **Robust Performance:** Outperforms the BSM baseline in mean P&L (-10.73 vs -10.89) by intelligently managing rebalancing frequency.

## 🛠️ Methodology

The framework formulates hedging as a Markov Decision Process (MDP):

### 1. Market Environment

* **Asset Dynamics:** Geometric Brownian Motion (GBM).
* **Instrument:** European Call Option.
* **Friction:** Linear transaction costs proportional to the value of shares traded.

### 2. Reinforcement Learning Configuration

* **Algorithm:** Deep Deterministic Policy Gradient (DDPG) (Actor-Critic).
* **State Space ():** `[Time to Maturity, Moneyness (S/K), Previous Position]`.
* *Note: Including the previous position allows the agent to calculate marginal transaction costs.*


* **Action Space ():** Continuous hedge ratio .
* **Reward Function:**


* Penalizes hedging error, transaction costs, and large deviations from the theoretical Delta.



## 📊 Performance

Evaluated over 5,000 training episodes and 1,000 out-of-sample test episodes:

| Metric | Neuro-Hedge (RL) | Black-Scholes (Baseline) |
| --- | --- | --- |
| **Mean P&L** | **-10.73** | -10.89 |
| **Transaction Costs** | ~0.50 | ~0.44 |
| **Strategy Style** | Smoothed / Adaptive | Continuous Rebalancing |

*The RL agent incurs slightly higher costs but achieves better overall P&L by avoiding unnecessary trades that do not significantly reduce risk.*

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neuro-hedge.git
cd neuro-hedge

# Install dependencies
pip install -r requirements.txt

```

## 💻 Usage

To train the agent using the guided curriculum:

```python
# Example: Train the Guided DDPG Agent
python train.py \
    --episodes 5000 \
    --volatility 0.2 \
    --cost_rate 0.002 \
    --teacher_forcing True

```

To evaluate the trained model against the Black-Scholes baseline:

```python
python evaluate.py --model_path models/best_actor.pth

```
