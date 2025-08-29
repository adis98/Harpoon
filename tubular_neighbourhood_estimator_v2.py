import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm
from generate_mask import generate_mask
from model import MLPDiffusion
from dataset import Preprocessor, get_eval
from utils import calc_diffusion_hyperparams
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--timesteps', type=int, default=200, help='Number of diffusion steps.')
parser.add_argument('--beta_0', type=float, default=0.0001, help='initial variance schedule')
parser.add_argument('--beta_T', type=float, default=0.02, help='last variance schedule')
parser.add_argument('--mask', type=str, default='MAR', help='Masking mechanisms.')
parser.add_argument('--num_trials', type=int, default=5, help='Number of sampling times.')
parser.add_argument('--ratio', type=str, default="0.25", help='Masking ratio.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    dataname = args.dataname
    device = args.device
    hid_dim = args.hid_dim
    mask_type = args.mask
    ratio = float(args.ratio)
    num_trials = args.num_trials
    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('OHE', prepper.df_train)
    test_X = prepper.encodeDf('OHE', prepper.df_test)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (
    np.concatenate((np.mean(train_X[:, :num_numeric], axis=0), np.zeros(train_X.shape[1] - num_numeric)), axis=0),
    np.concatenate((np.std(train_X[:, :num_numeric], axis=0), np.ones(train_X.shape[1] - num_numeric)), axis=0))
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    X = torch.tensor(X)
    diffusion_config = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_T)
    models_dir = f'saved_models/{args.dataname}/'
    model_path = os.path.join(models_dir, "diffputer_selfmade.pt")
    model = MLPDiffusion(in_dim, hid_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    num_samples = 5
    sample_indices = np.random.choice(X.shape[0], size=num_samples, replace=False)
    trial_condition_nrs = []
    trial_cosine_sims = []
    for idx in sample_indices:
        mean_cosine_sims = []
        condition_nrs = []
        element = X[idx, :].reshape((1, -1))
        X_batch = element.repeat(150, 1).to(device, dtype=torch.float32)
        with torch.no_grad():
            for t in range(args.timesteps - 1, -1, -1):
                timesteps = torch.full(size=(X_batch.shape[0],), fill_value=t).to(device)
                alpha_t = diffusion_config['Alpha'][t].to(device)
                alpha_bar_t = diffusion_config['Alpha_bar'][t].to(device)
                alpha_bar_t_1 = diffusion_config['Alpha_bar'][t - 1].to(device) if t >= 1 else torch.tensor(1).to(
                    device)
                sigmas = torch.normal(0, 1, size=X_batch.shape).to(device)
                """Forward noising"""
                coeff_1 = torch.sqrt(alpha_bar_t)
                coeff_2 = torch.sqrt(1 - alpha_bar_t)
                batch_noised = coeff_1 * X_batch + coeff_2 * sigmas
                batch_noised = batch_noised.to(device)
                sigmas_predicted = model(batch_noised, timesteps)
                x_0_hats = (batch_noised - torch.sqrt(1-alpha_bar_t) * sigmas_predicted)/torch.sqrt(alpha_bar_t)
                vectors = x_0_hats - batch_noised
                normed_vectors = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-12)
                C = torch.matmul(normed_vectors, normed_vectors.T)
                mean_cossim = (C.sum() - C.trace()) / (C.numel() - C.shape[0])
                _, S, _ = torch.linalg.svd(normed_vectors, full_matrices=False)
                condition_nr = S[0]/(S[-1] + 1e-12)
                condition_nrs.append(condition_nr.cpu().numpy())
                mean_cosine_sims.append(mean_cossim.cpu().numpy())
        condition_nrs = np.array(condition_nrs)
        mean_cosine_sims = np.array(mean_cosine_sims)
        trial_condition_nrs.append(condition_nrs)
        trial_cosine_sims.append(mean_cosine_sims)
    trial_condn_nrs = np.array(trial_condition_nrs)
    trial_cosine_sims = np.array(trial_cosine_sims)
    avg_condn_nrs = np.mean(trial_condn_nrs, axis=0)
    std_condn_nrs = np.std(trial_condn_nrs, axis=0)
    avg_cosine_sims = np.mean(trial_cosine_sims, axis=0)
    std_cosine_sims = np.std(trial_cosine_sims, axis=0)
    Ts = np.arange(args.timesteps-1, -1, -1)

    # Create the plot
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))  # 1 row, 2 columns
    ax = axes[0]

    # Plot the mean line
    ax.plot(Ts, avg_condn_nrs, color='green', label='Mean')

    # Add the "cloud" of standard deviation
    ax.fill_between(
        Ts,
        avg_condn_nrs - std_condn_nrs,
        avg_condn_nrs + std_condn_nrs,
        color='green',
        alpha=0.2,
        label='±1 Std Dev'
    )

    ax.invert_xaxis()
    # Labels and title
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Avg. condition number")
    ax.set_title(f"Avg. condition numbers of unit vectors pointing from a noised sample\n to their predicted denoised sample over 5 trials: {args.dataname}")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    # Plot the mean line
    ax.plot(Ts, avg_cosine_sims, color='orange', label='Mean')

    # Add the "cloud" of standard deviation
    ax.fill_between(
        Ts,
        avg_cosine_sims - std_cosine_sims,
        avg_cosine_sims + std_cosine_sims,
        color='orange',
        alpha=0.2,
        label='±1 Std Dev'
    )

    ax.invert_xaxis()
    # Labels and title
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Avg. cosine similarity with std. dev cloud")
    ax.set_title(f"Avg. cosine similarities of unit vectors pointing from a noised\n sample to their predicted denoised sample over 5 trials: {args.dataname}")
    ax.legend()
    ax.grid(True)
    plt.savefig(f'experiments/tubular_region_plots/{args.dataname}.pdf')




