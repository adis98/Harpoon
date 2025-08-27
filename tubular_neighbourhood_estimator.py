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
    MSEs, STDs, COSSIMs, STDCOSSIMs = [], [], [], []
    X_batch = X[:1024, :].to(device, dtype=torch.float32)
    with torch.no_grad():
        for t in range(args.timesteps - 1, 9, -1):
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
            per_sample_mse = torch.mean((sigmas_predicted - sigmas)**2, dim=1)
            per_sample_cosine_similarity = torch.sum((sigmas_predicted * sigmas), dim=1)/(torch.sqrt(torch.sum((sigmas_predicted**2), dim=1)) * torch.sqrt(torch.sum((sigmas**2), dim=1)))
            avg_mse = torch.mean(per_sample_mse)
            std_dev_mse = torch.std(per_sample_mse)
            avg_cossim = torch.mean(per_sample_cosine_similarity)
            std_dev_cossim = torch.std(per_sample_cosine_similarity)
            COSSIMs.append(avg_cossim.cpu().numpy())
            STDCOSSIMs.append(std_dev_cossim.cpu().numpy())
            MSEs.append(avg_mse.cpu().numpy())
            STDs.append(std_dev_mse.cpu().numpy())

    MSEs = np.array(MSEs)
    STDs = np.array(STDs)
    COSSIMs = np.array(COSSIMs)
    STDCOSSIMs = np.array(STDCOSSIMs)
    Ts = np.arange(199, 9, -1)

    # Create the plot
    plt.figure(figsize=(8, 5))

    # Plot the mean line
    plt.plot(Ts, MSEs, color='green', label='Mean')

    # Add the "cloud" of standard deviation
    plt.fill_between(
        Ts,
        MSEs - STDs,
        MSEs + STDs,
        color='green',
        alpha=0.2,
        label='±1 Std Dev'
    )

    plt.gca().invert_xaxis()
    # Labels and title
    plt.xlabel("Denoising step")
    plt.ylabel("MSE with std. dev cloud")
    plt.title(f"Avg. MSE between sampled noise vector and model estimate: {args.dataname}")
    plt.legend()
    plt.grid(True)

    plt.show()

    plt.figure(figsize=(8, 5))

    # Plot the mean line
    plt.plot(Ts, COSSIMs, color='orange', label='Mean')

    # Add the "cloud" of standard deviation
    plt.fill_between(
        Ts,
        COSSIMs - STDCOSSIMs,
        COSSIMs + STDCOSSIMs,
        color='orange',
        alpha=0.2,
        label='±1 Std Dev'
    )

    plt.gca().invert_xaxis()
    # Labels and title
    plt.xlabel("Denoising step")
    plt.ylabel("Avg. cosine similarity with std. dev cloud")
    plt.title(f"Avg. cosine similarity between sampled noise and model estimate: {args.dataname}")
    plt.legend()
    plt.grid(True)

    plt.show()



