import numpy as np
import pandas as pd
import eval.simulation_eval as sim_eval
import eval.mean_cov as mean_cov
from pathlib import Path
import os
from natsort import natsorted

## Ground Truth Data
training_data_path = "simulation_experiment_data/training_data.npy"
ground_truth_mean_path = "simulation_experiment_data/ground_truth_mean.npy"
ground_truth_cov_path = "simulation_experiment_data/ground_truth_cov.npy"
training_data = np.load(training_data_path)
training_data = training_data.reshape(training_data.shape[0], -1)  # Reshape to (num_samples, asset_dim)
ground_truth_mean = np.load(ground_truth_mean_path)
ground_truth_cov = np.load(ground_truth_cov_path)

## Training Sample Data
train_sizes = [512, 1024, 2048, 4096, 8192]
training_data_dict = {}
for num in train_sizes:
    training_data_dict[num] = training_data[:num]

## Sythetic Sample Data
def load_training_data(folder_path, extension=".npy", verbose=False):
    """
    从指定文件夹加载所有特定后缀的 numpy 文件并合并。

    参数:
    - folder_path: 文件夹路径 (字符串或 Path 对象)
    - extension: 文件扩展名，默认为 ".npy"
    - verbose: 是否打印加载的文件顺序

    返回:
    - 合并后的 np.ndarray
    """
    path = Path(folder_path)

    # 检查路径是否存在
    if not path.exists():
        raise FileNotFoundError(f"找不到文件夹: {folder_path}")

    # 获取并按自然顺序排序文件
    files = natsorted(path.glob(f"*{extension}"))
    files = [f for f in files if f.is_file()]

    if verbose:
        print(f"找到 {len(files)} 个文件，加载顺序:")
        for i, f in enumerate(files[:5]):  # 只显示前5个
            print(f"  {i + 1}: {f.name}")
        print(f"  ... 共 {len(files)} 个文件")

    # 加载数据
    data_list = [np.load(f) for f in files]

    if not data_list:
        print(f" 警告: 在 {folder_path} 中没有找到 {extension} 文件。")
        return np.array([])

    # 合并数据
    return np.concatenate(data_list, axis=0)

## 采样数据的路径
synthetic_data_path = ['dfm_training_data_ts1772112605_seed42_num_examples512',
                       'dfm_training_data_ts1772111664_seed42_num_examples1024',
                       'dfm_training_data_ts1772108922_seed42_num_examples2048',
                       'dfm_training_data_ts1772104088_seed42_num_examples4096',
                       'dfm_training_data_ts1772104074_seed42_num_examples8192']
synthetic_data_dict = {}

## 加载采样数据
for file,num in zip(synthetic_data_path,train_sizes):
    file_path = os.path.join('samples', file)
    print(f"Loading synthetic data from: samples/{file} trained with {num} samples")
    synthetic_data_dict[num] = load_training_data(file_path,verbose=True)
    file_path = os.path.join(file_path,f'all/synthetic_data_all.npy')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 确保目录存在
    np.save(file_path, synthetic_data_dict[num])  # 保存合并后的数据到新的 .npy 文件


## 计算并保存ground-truth的eigenvalues和latent subspace
sim_eval.calculate_latent_subspace(ground_truth_cov, latent_dim=16, save_path='results', prefix=f'ground_truth')

for num in train_sizes:
    print(f"Calculating and saving eigenvalues and latent subspace for {num} training samples...")
    sim_eval.calculate_latent_subspace(mean_cov.calculate_mean_cov(synthetic_data_dict[num])[1], latent_dim=16, save_path='results', prefix=f'synthetic_{num}')
    sim_eval.calculate_latent_subspace(mean_cov.calculate_mean_cov(training_data_dict[num])[1], latent_dim=16, save_path='results', prefix=f'real_{num}')


    print(f"Calculating and saving eigenvalues relative errors for {num} training samples...")
    sim_eval.calculate_eigenvalues_relative_error(
        synthetic_sample_eigenvalues=np.load(f"results/synthetic_{num}_eigenvalues.npy"),
        real_sample_eigenvalues=np.load(f"results/real_{num}_eigenvalues.npy"),
        real_eigenvalues=np.load(f'results/ground_truth_eigenvalues.npy'),
        output_path=f"results/eigenvalues_error_{num}.npy"
    )

    print(f"Calculating and saving Frobenius norm errors for {num} training samples...")
    sim_eval.calculate_frobenius_norm_errors(
        synthetic_sample_latent_subspace=np.load(f"results/synthetic_{num}_latent_subspace.npy"),
        real_sample_latent_subspace=np.load(f"results/real_{num}_latent_subspace.npy"),
        real_latent_subspace=np.load(f'results/ground_truth_latent_subspace.npy'),
        output_path=f"results/frobenius_error_{num}.npy"
    )

    print(f"Calculating and saving l^2 relative errors of the mean for {num} training samples...")
    sim_eval.calculate_frobenius_norm_errors(mean_cov.calculate_mean_cov(synthetic_data_dict[num])[0],
                                            mean_cov.calculate_mean_cov(training_data_dict[num])[0],
                                            ground_truth_mean,
                                            output_path=f"results/mean_l2_error_{num}.npy")

    print(f"Calculating and saving frobenius relative errors of the covariance for {num} training samples...")
    sim_eval.calculate_frobenius_norm_errors(mean_cov.calculate_mean_cov(synthetic_data_dict[num])[1],
                                            mean_cov.calculate_mean_cov(training_data_dict[num])[1],
                                            ground_truth_cov,
                                            output_path=f"results/cov_fro_error_{num}.npy")
# ----------------------------
# 1) Table 1: aggregate errors (eigenvalues_error + frobenius_error)
# ----------------------------

table_dir = Path('results') / 'table1'
table_dir.mkdir(parents=True, exist_ok=True)

rows = []
for num in train_sizes:
    eig_obj = np.load(f"results/eigenvalues_error_{num}.npy", allow_pickle=True).item()
    fro_obj = np.load(f"results/frobenius_error_{num}.npy", allow_pickle=True).item()

    rows.append(
        {
            "train_size": int(num),
            "eigen_synth_error": float(eig_obj.get("synthetic_error")),
            "eigen_real_error": float(eig_obj.get("real_error")),
            "eigen_ratio": float(eig_obj.get("ratio")),
            "fro_synth_error": float(fro_obj.get("synthetic_error")),
            "fro_real_error": float(fro_obj.get("real_error")),
            "fro_ratio": float(fro_obj.get("ratio")),
        }
    )

df_table1 = pd.DataFrame(rows).sort_values("train_size")
df_table1.to_csv(table_dir / "table1_errors.csv", index=False)
print(f"Saved Table 1 -> {table_dir / 'table1_errors.csv'}")

# ----------------------------
# 2) Figure1: use correct indices (max/min std, max/min mean) and save to results/figure
# ----------------------------
from scipy.stats import norm
import matplotlib.pyplot as plt

fig_dir = Path('results') / 'figure'
fig_dir.mkdir(parents=True, exist_ok=True)

# pick indices based on ground truth training data distribution
asset_means = np.mean(training_data, axis=0)
asset_stds = np.std(training_data, axis=0)

idx_map = {
    "max_std": int(np.argmax(asset_stds)),
    "min_std": int(np.argmin(asset_stds)),
    "max_mean": int(np.argmax(asset_means)),
    "min_mean": int(np.argmin(asset_means)),
}

training_arr = np.load(training_data_path)
training_arr = training_arr.reshape(training_arr.shape[0], -1)

generated_path = 'samples/dfm_training_data_ts1772108922_seed42_num_examples2048/all/synthetic_data_all.npy'
generated_arr = np.load(generated_path)
generated_arr = generated_arr.reshape(generated_arr.shape[0], -1)

bins_num = 100
for tag, idx in idx_map.items():
    # 合理范围：用训练+生成的合并分布估计mu和sigma，设置 x_bound = mu ± 3*sigma（但不小于1.0，不大于10.0）
    combined = np.concatenate([training_arr[:, idx], generated_arr[:, idx]], axis=0)
    mu = float(np.mean(combined))
    sig = float(np.std(combined) + 1e-12)
    x_bound = max(1.0, min(10.0, abs(mu) + 3.0 * sig))
    print(f"[{tag}] idx={idx} | x_bound={x_bound:.3f} y_bound=0.05 zoom=0")

    fig, axes, _ = sim_eval.comparision_histplot_simulation(
        idx,
        training_data_path,
        generated_path,
        ground_truth_mean,
        ground_truth_cov,
        bins_num,
        int(np.ceil(x_bound)),
        0.05,
        0,
        show=False,
    )

    # 保存
    out_path = fig_dir / f"hist_{tag}_idx{idx}.png"
    fig.savefig(out_path, dpi=800, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure -> {out_path}")

print(f"All figures saved under: {fig_dir}")


# 3） Table2
table_dir = Path('results') / 'table2'
table_dir.mkdir(parents=True, exist_ok=True)

rows = []
for num in train_sizes:
    mean_obj = np.load(f"results/mean_l2_error_{num}.npy", allow_pickle=True).item()
    cov_obj = np.load(f"results/cov_fro_error_{num}.npy", allow_pickle=True).item()

    rows.append(
        {
            "train_size": int(num),
            "mean_synth_error": float(mean_obj.get("synthetic_error")),
            "mean_real_error": float(mean_obj.get("real_error")),
            "mean_ratio": float(mean_obj.get("ratio")),
            "cov_synth_error": float(cov_obj.get("synthetic_error")),
            "cov_real_error": float(cov_obj.get("real_error")),
            "cov_ratio": float(cov_obj.get("ratio")),
        }
    )

df_table1 = pd.DataFrame(rows).sort_values("train_size")
df_table1.to_csv(table_dir / "table2_errors.csv", index=False)
print(f"Saved Table 2 -> {table_dir / 'table2_errors.csv'}")