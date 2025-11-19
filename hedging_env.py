import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from market_simulator import GBMSimulator
from black_scholes import EuropeanCallOption, BlackScholes

class HedgingEnv(gym.Env):
    """
    FIXED hedging environment with Normalized Inputs and Squared Error Reward.
    """
    
    def __init__(self, simulator_params, option_params, risk_free_rate, transaction_cost_rate):
        super(HedgingEnv, self).__init__()
        
        self.S0 = simulator_params['S0']
        self.mu = simulator_params['mu']
        self.sigma = simulator_params['sigma']
        self.T = simulator_params['T']
        self.dt = simulator_params['dt']
        
        self.K = option_params['K']
        
        self.r = risk_free_rate
        self.c_rate = transaction_cost_rate
        
        # Penalty factor for transaction costs
        self.lambda_txn = 0.01
        
        self.simulator = GBMSimulator(self.S0, self.mu, self.sigma, self.T, self.dt)
        self.option = EuropeanCallOption(self.K, self.T)
        
        self.n_steps = self.simulator.n_steps
        self.current_step = 0
        self.stock_path = []
        self.current_stock_price = 0.0
        
        self.cash = 0.0
        self.num_shares = 0.0
        self.total_cost = 0.0
        self.time = 0.0
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Update shape to 3 to include num_shares
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([self.T, np.inf, 1.0], dtype=np.float32), 
            shape=(3,), 
            dtype=np.float32
        )
        
        self.initial_cash = BlackScholes.call_price(self.S0, self.K, self.T, self.r, self.sigma)

    def _get_obs(self):
        time_to_expiry = self.T - self.time
        normalized_price = self.current_stock_price / self.K
        
        # Include current position so agent can manage transaction costs
        obs = np.array([time_to_expiry, normalized_price, self.num_shares], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.stock_path = self.simulator.simulate_path()
        self.current_step = 0
        self.time = 0.0
        self.current_stock_price = self.stock_path[0]
        
        self.cash = self.initial_cash
        
        # Start with Black-Scholes delta position
        bs_delta = BlackScholes.call_delta(self.current_stock_price, self.K, self.T, self.r, self.sigma)
        self.num_shares = bs_delta
        self.cash -= bs_delta * self.current_stock_price
        
        self.total_cost = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        new_num_shares = action[0]
        shares_to_trade = new_num_shares - self.num_shares
        
        transaction_cost = abs(shares_to_trade) * self.current_stock_price * self.c_rate
        
        cash_flow = -shares_to_trade * self.current_stock_price
        self.cash += cash_flow
        self.cash -= transaction_cost
        self.total_cost += transaction_cost
        
        self.num_shares = new_num_shares
        
        self.current_step += 1
        self.time = self.current_step * self.dt
        
        self.cash *= (1 + self.r * self.dt)
        
        self.current_stock_price = self.stock_path[self.current_step]
        
        terminated = (self.current_step == self.n_steps)
        
        # --- Reward Calculation ---
        portfolio_value = self.num_shares * self.current_stock_price + self.cash
        
        time_to_expiry = self.T - self.time
        if time_to_expiry > 0:
            option_value = BlackScholes.call_price(
                self.current_stock_price, self.K, time_to_expiry, self.r, self.sigma
            )
        else:
            option_value = self.option.payoff(self.current_stock_price)
        
        hedging_error = portfolio_value - option_value - self.initial_cash
        
        # Use Squared Error.
        reward = -(hedging_error**2)
        
        # Secondary penalty for transaction costs
        reward -= (self.lambda_txn * transaction_cost)
        
        # FIX: Removed reward normalization (division by 10.0) to boost training signal.
        
        info = {}
        
        if terminated:
            self.cash += self.num_shares * self.current_stock_price
            option_payoff = self.option.payoff(self.current_stock_price)
            self.cash -= option_payoff
            
            terminal_pnl = self.cash - self.initial_cash
            info['terminal_pnl'] = terminal_pnl
            
        return self._get_obs(), reward, terminated, False, info