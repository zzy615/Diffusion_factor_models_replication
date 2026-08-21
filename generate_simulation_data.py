import numpy as np
from diffusion_factor_model.diffusion_factor_model import GaussianLatentSampler2D_Finance

class SimulationDataGenerator:
    def __init__(self,asset_dim, latent_dim, num_samples, random_state, imag_size):
        self.asset_dim = asset_dim
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.imag_size = imag_size

    def generate_synthetic_data(self):
        # Generate the latent factors
        mu_F = self.rng.uniform(0,0.1, self.latent_dim)
        cov_F = np.diag((1.5*mu_F)**2)

        # Generate the idiosyncratic noise
        sigma_epsilon = self.rng.uniform(0, 0.4, self.asset_dim)
        mu_epsilon = np.zeros(self.asset_dim)
        cov_epsilon = np.diag(sigma_epsilon**2)

        # Generate the returns using the factor model
        factor_sampler = GaussianLatentSampler2D_Finance(self.latent_dim, self.imag_size,self.random_state)
        factor,returns,ground_truth_mean,ground_truth_cov = factor_sampler.generate_data(self.num_samples, mu_F, cov_F, mu_epsilon, cov_epsilon)

        return returns, ground_truth_mean, ground_truth_cov

    def describe_data(self, returns, n_show):
        print("Original Returns shape:", returns.shape)
        if returns.ndim != 2:
            returns = returns.reshape(returns.shape[0], -1)  # Reshape to (num_samples, asset_dim)
            print("Reshaped Returns shape:", returns.shape)
        assets_mean = np.mean(returns, axis=0)
        assets_std = np.std(returns, axis=0)

        def stats_summary(arr):
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "25%": float(np.percentile(arr, 25)),
                "50%": float(np.percentile(arr, 50)),
                "75%": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
            }

        mean_stats = stats_summary(assets_mean)
        std_stats = stats_summary(assets_std)

        print("\nCross-sectional stats for per-asset mean:")
        print(f"  mean={mean_stats['mean']:.6f}, std={mean_stats['std']:.6f}, min={mean_stats['min']:.6f}, 25%={mean_stats['25%']:.6f}, 50%={mean_stats['50%']:.6f}, 75%={mean_stats['75%']:.6f}, max={mean_stats['max']:.6f}")

        print("\nCross-sectional stats for per-asset std (volatility):")
        print(f"  mean={std_stats['mean']:.6f}, std={std_stats['std']:.6f}, min={std_stats['min']:.6f}, 25%={std_stats['25%']:.6f}, 50%={std_stats['50%']:.6f}, 75%={std_stats['75%']:.6f}, max={std_stats['max']:.6f}")

        # 显示若干资产的具体 mean/std（按资产索引）
        n_show = min(n_show, asset_dim)
        print(f"\nExample per-asset mean & std for first {n_show} assets:")
        for i in range(n_show):
            print(f"  asset {i}: mean={assets_mean[i]:.6f}, std={assets_std[i]:.6f}")

        # 显示均值最高和最低的若干资产索引（可快速看横截面差异）
        top_mean_idx = np.argsort(assets_mean)[-n_show:][::-1]
        bottom_mean_idx = np.argsort(assets_mean)[:n_show]
        print(f"\nTop {n_show} assets by mean:")
        for i in top_mean_idx:
            print(f"  asset {i}: mean={assets_mean[i]:.6f}, std={assets_std[i]:.6f}")
        print(f"\nBottom {n_show} assets by mean:")
        for i in bottom_mean_idx:
            print(f"  asset {i}: mean={assets_mean[i]:.6f}, std={assets_std[i]:.6f}")

        # 显示均值最高和最低的若干资产索引（可快速看横截面差异）
        top_std_idx = np.argsort(assets_std)[-n_show:][::-1]
        bottom_std_idx = np.argsort(assets_std)[:n_show]
        print(f"\nTop {n_show} assets by std:")
        for i in top_std_idx:
            print(f"  asset {i}: mean={assets_mean[i]:.6f}, std={assets_std[i]:.6f}")
        print(f"\nBottom {n_show} assets by std:")
        for i in bottom_std_idx:
            print(f"  asset {i}: mean={assets_mean[i]:.6f}, std={assets_std[i]:.6f}")

    def save_data(self, returns, ground_truth_mean, ground_truth_cov):
        np.save('simulation_experiment_data/training_data.npy', returns)
        np.save('simulation_experiment_data/ground_truth_mean.npy', ground_truth_mean)
        np.save('simulation_experiment_data/ground_truth_cov.npy', ground_truth_cov)


if __name__ == "__main__":
    asset_dim = 2048
    latent_dim = 16
    num_samples = 8192
    random_state = 42
    imag_size = (32,64)
    print("Generating synthetic data...")
    generator = SimulationDataGenerator(asset_dim, latent_dim, num_samples, random_state, imag_size)
    returns, ground_truth_mean,ground_truth_cov = generator.generate_synthetic_data()
    print("Saving synthetic data...")
    generator.save_data(returns, ground_truth_mean, ground_truth_cov)
    print("Data generation and saving completed.")
    generator.describe_data(returns, n_show=5)
