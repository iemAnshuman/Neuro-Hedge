import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from market_simulator import GBMSimulator
from black_scholes import EuropeanCallOption, BlackScholes

class HedgingEnv(gym.Env):
    """
    FIXED hedging environment with proper reward structure
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
        
        # FIXED: Much lower transaction cost penalty
        self.lambda_txn = 0.001
        
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
        
        # FIXED: Remove current_num_shares from observation
        # Agent should not see its current position - prevents "do nothing" learning
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32), 
            high=np.array([self.T, np.inf], dtype=np.float32), 
            shape=(2,), 
            dtype=np.float32
        )
        
        self.initial_cash = BlackScholes.call_price(self.S0, self.K, self.T, self.r, self.sigma)

    def _get_obs(self):
        time_to_expiry = self.T - self.time
        # FIXED: Only return time and price, not current position
        obs = np.array([time_to_expiry, self.current_stock_price], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.stock_path = self.simulator.simulate_path()
        self.current_step = 0
        self.time = 0.0
        self.current_stock_price = self.stock_path[0]
        
        self.cash = self.initial_cash
        
        # FIXED: Start with Black-Scholes delta position instead of 0
        bs_delta = BlackScholes.call_delta(self.current_stock_price, self.K, self.T, self.r, self.sigma)
        self.num_shares = bs_delta
        self.cash -= bs_delta * self.current_stock_price  # Pay for initial position
        
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
        
        # Calculate the portfolio value change for reward
        # This is the change in hedging portfolio value
        portfolio_value = self.num_shares * self.current_stock_price + self.cash
        
        # Calculate what the option is worth now
        time_to_expiry = self.T - self.time
        if time_to_expiry > 0:
            option_value = BlackScholes.call_price(
                self.current_stock_price, self.K, time_to_expiry, self.r, self.sigma
            )
        else:
            option_value = self.option.payoff(self.current_stock_price)
        
        # The hedging error is the difference between our hedge portfolio and option value
        # We want to minimize this
        hedging_error = abs(portfolio_value - option_value - self.initial_cash)
        
        # FIXED: Better reward structure
        # Primary goal: minimize hedging error
        # Secondary: minimize transaction costs
        reward = -hedging_error - (self.lambda_txn * transaction_cost)
        
        # Normalize reward to reasonable scale
        reward = reward / self.S0
        
        info = {}
        
        if terminated:
            # Calculate final P&L
            self.cash += self.num_shares * self.current_stock_price
            option_payoff = self.option.payoff(self.current_stock_price)
            self.cash -= option_payoff
            
            terminal_pnl = self.cash - self.initial_cash
            info['terminal_pnl'] = terminal_pnl
            
        return self._get_obs(), reward, terminated, False, info