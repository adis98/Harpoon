import os
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm

from model import MLPDiffusion, Model
from dataset import load_dataset_nomask, get_eval, mean_std_nomask
from diffusion_utils import sample_step, impute_mask

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'

if __name__ == '__main__':

    dataname = args.dataname
    device = args.device
    hid_dim = args.hid_dim
    num_steps = args.num_steps
    train_X, test_X, train_num, test_num, train_cat_idx, test_cat_idx, cat_bin_num = load_dataset_nomask(
        dataname)
    mean_X, std_X = mean_std_nomask(train_X)
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X / 2
    X = torch.tensor(X)
    X_test = (test_X - mean_X) / std_X / 2
    X_test = torch.tensor(X_test)
    MAEs = []
    RMSEs = []

    MAEs_out = []
    RMSEs_out = []

    start_time = time.time()
    models_dir = f'saved_models/{args.dataname}/'
    os.makedirs(f'{models_dir}') if not os.path.exists(f'{models_dir}') else None
    batch_size = 4096
    train_loader = DataLoader(
        X,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    num_epochs = 1000+1

    denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

    print(denoise_fn)

    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, verbose=False)

    model.train()

    best_loss = float('inf')
    patience = 0

    # progress bar
    pbar = tqdm(range(num_epochs), desc='Training')
    for epoch in pbar:

        batch_loss = 0.0
        len_input = 0

        for batch in train_loader:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{models_dir}/diffputer.pt')
        else:
            patience += 1
            if patience == 50:
                print('Early stopping')
                break

        pbar.set_postfix(loss=curr_loss)

        # if epoch % 1000 == 0:
        #     torch.save(model.state_dict(), f'{ckpt_dir}/{iteration}/model_{epoch}.pt')

    end_time = time.time()

    ## E-Step: Missing Value Imputation

    # In-sample imputation

    rec_Xs = []

    # for trial in tqdm(range(num_trials), desc='In-sample imputation'):
    #     X_miss = (1. - mask_train.float()) * X
    #     X_miss = X_miss.to(device)
    #     impute_X = X_miss
    #
    #     in_dim = X.shape[1]
    #
    #     denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
    #
    #     model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    #     model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt'))
    #
    #     # ==========================================================
    #
    #     net = model.denoise_fn_D
    #
    #     num_samples, dim = X.shape[0], X.shape[1]
    #     rec_X = impute_mask(net, impute_X, mask_train, num_samples, dim, num_steps, device)
    #
    #     mask_int = mask_train.to(torch.float).to(device)
    #     rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
    #     rec_Xs.append(rec_X)
    #
    # rec_X = torch.stack(rec_Xs, dim=0).mean(0)
    #
    # rec_X = rec_X.cpu().numpy() * 2
    # X_true = X.cpu().numpy() * 2
    #
    # np.save(f'{ckpt_dir}/iter_{iteration + 1}.npy', rec_X)
    #
    # pred_X = rec_X[:]
    # len_num = train_num.shape[1]
    #
    # res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
    # pred_X[:, len_num:] = res
    #
    # mae, rmse = get_eval(dataname, pred_X, X_true, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
    # MAEs.append(mae)
    # RMSEs.append(rmse)
    #
    # print('in-sample', mae, rmse)
    #
    # # out-of_sample_imputation
    #
    # rec_Xs = []
    #
    # for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
    #     # For out-of-sample imputation, no results from previous iterations are used
    #
    #     X_miss = (1. - mask_test.float()) * X_test
    #     X_miss = X_miss.to(device)
    #     impute_X = X_miss
    #
    #     in_dim = X_test.shape[1]
    #
    #     denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
    #
    #     model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    #     model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt'))
    #
    #     # ==========================================================
    #     net = model.denoise_fn_D
    #
    #     num_samples, dim = X_test.shape[0], X_test.shape[1]
    #     rec_X = impute_mask(net, impute_X, mask_test, num_samples, dim, num_steps, device)
    #
    #     mask_int = mask_test.to(torch.float).to(device)
    #     rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
    #     rec_Xs.append(rec_X)
    #
    # rec_X = torch.stack(rec_Xs, dim=0).mean(0)
    #
    # rec_X = rec_X.cpu().numpy() * 2
    # X_true = X_test.cpu().numpy() * 2
    #
    # pred_X = rec_X[:]
    # len_num = train_num.shape[1]
    # res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
    # pred_X[:, len_num:] = res
    #
    # mae_out, rmse_out = get_eval(dataname, pred_X, X_true, test_cat_idx, test_num.shape[1], cat_bin_num,
    #                              ori_test_mask, oos=True)
    # MAEs_out.append(mae_out)
    # RMSEs_out.append(rmse_out)
    #
    # result_save_path = f'results/{dataname}/rate{ratio}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
    # os.makedirs(result_save_path) if not os.path.exists(result_save_path) else None
    #
    # with open(f'{result_save_path}/result.txt', 'a+') as f:
    #
    #     f.write(f'iteration {iteration}, MAE: in-sample: {mae}, out-of-sample: {mae_out} \n')
    #     f.write(f'iteration {iteration}: RMSE: in-sample: {rmse}, out-of-sample: {rmse_out} \n')
    #
    # print('out-of-sample', mae_out, rmse_out)
    #
    # print(f'saving results to {result_save_path}')
