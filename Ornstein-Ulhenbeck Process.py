import numpy as np
import matplotlib.pyplot as plt

class OrnsteinUhlenbeckProcess:
    # O-U process approximation
    # dx = beta*(mu - x)*dt + sigma*dW
    def __init__(self,x0 = None,mu = 0.0,beta = 0.1,dt = 0.01,sigma = 1,random_seed = 824):
        self.mu = mu
        self.beta = beta
        self.dt = dt
        self.sigma = sigma
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.x0 = x0
        self.reset()


    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.beta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def __repr__(self):
        return f'OrnsteinUhlenbeckProcess(mu={self.mu}, beta={self.beta}, dt={self.dt}, sigma={self.sigma})'


if __name__ == "__main__":
    ou_process = OrnsteinUhlenbeckProcess(mu=np.array([0.0]), beta=0.15, dt=0.01, sigma=0.2,random_seed=42)
    num_steps = 1000
    samples = np.zeros((num_steps, 1))
    for i in range(num_steps):
        samples[i] = ou_process()

    plt.plot(samples)
    plt.title('Ornstein-Uhlenbeck Process Sample Path')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid()
    plt.show()

    print(ou_process)
    print("First 10 samples:", samples[-10:])
