import numpy as np
import math

class GBMSimulator:
    
    def __init__(self, S0, mu, sigma, T, dt):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.rng = np.random.default_rng()

    def simulate_path(self):
        path = np.zeros(self.n_steps + 1)
        path[0] = self.S0
        
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt)
        
        Z = self.rng.standard_normal(self.n_steps)
        
        for i in range(1, self.n_steps + 1):
            path[i] = path[i-1] * math.exp(drift + diffusion * Z[i-1])
            
        return path

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    dt = 1/252
    
    simulator = GBMSimulator(S0, mu, sigma, T, dt)
    
    plt.figure(figsize=(10, 6))
    for _ in range(10):
        path = simulator.simulate_path()
        plt.plot(path)
        
    plt.title('GBM Path Simulations (10 Paths)')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.show()