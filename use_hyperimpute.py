import os
import sys
import numpy as np
import pandas as pd
import argparse
import warnings
from tqdm import tqdm
import torch
from hyperimpute.plugins.imputers import Imputers, ImputerPlugin
from hyperimpute.plugins.utils.metrics import RMSE
from dataset import Preprocessor, get_eval
from generate_mask import generate_mask

from sklearn import metrics
import xgboost as xgb
from hyperimpute.utils.distributions import enable_reproducible_results
from hyperimpute.plugins.utils.simulate import simulate_nan

# enable_reproducible_results()
warnings.filterwarnings('ignore')

# def parse_args():
parser = argparse.ArgumentParser(description='Train HyperImpute on tabular datasets')
parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--mask', type=str, default='MCAR', help='Masking mechanism: MCAR, MAR, MNAR_logistic_T2')
parser.add_argument('--ratio', type=float, default="0.25", help='Missing ratio')
parser.add_argument('--num_trials', type=int, default=5, help='Number of mask trials')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

args = parser.parse_args()
# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'
metrics_headers = ["Seed", "AUROC"]
test_score = []

def masked_mse(X_pred, X_true, mask, mask_marks_missing=True):
    # 只算数值列；mask==True 的位置才计入
    num_cols = X_true.select_dtypes(include=[np.number]).columns
    Xp = X_pred[num_cols].to_numpy()
    Xt = X_true[num_cols].to_numpy()
    M  = mask[num_cols].astype(bool).to_numpy()
    if not mask_marks_missing:
        M = ~M
    return float(np.nanmean(((Xp - Xt) ** 2)[M]))

import numpy as np
import pandas as pd

def masked_mse1(X_pred, X_true, mask, mask_marks_missing=True):
    """Compute MSE on numeric columns only, restricted to mask==True cells.
       Returns np.nan if there are no numeric columns or no selected cells.
    """
    # Ensure DataFrames and align shapes/columns
    X_true = pd.DataFrame(X_true)
    X_pred = pd.DataFrame(X_pred).reindex_like(X_true)
    mask   = pd.DataFrame(mask).reindex_like(X_true)

    # Numeric columns only
    num_cols = X_true.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return np.nan

    # Normalize mask to boolean; NaN -> False
    M = mask[num_cols]
    M = (M.astype(bool)) & M.notna()
    M = M.to_numpy()

    if mask_marks_missing is False:
        M = ~M

    # If nothing is selected, return nan
    if not M.any():
        return np.nan

    Xp = X_pred[num_cols].to_numpy()
    Xt = X_true[num_cols].to_numpy()

    diff2 = (Xp - Xt) ** 2
    return float(np.nanmean(diff2[M]))

def ampute(x, mechanism, p_miss):
    x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]
    # print(x_miss, mask)
    # return pd.DataFrame(x), pd.DataFrame(x_miss, columns = x.columns), pd.DataFrame(mask, columns = x.columns)
    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)


def main():
    # Parse command line arguments
    # args = parse_args()
    
    # torch.manual_seed(42)
    np.random.seed(42)
    dataname = args.dataname
    device = args.device
    # hid_dim = args.hid_dim
    mask_type = args.mask
    ratio = float(args.ratio)
    num_trials = args.num_trials

    
    ampute_mechanism = mask_type
    if ampute_mechanism == 'MNAR':
        ampute_mechanism = 'MNAR_logistic_T2'
    p_miss = ratio

    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    print(f"Dataset: {dataname}, Mask: {mask_type}, Ratio: {ratio}, Trials: {num_trials}")  
    
    # Load and preprocess the dataset
    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('OHE', prepper.df_train)
    test_X = prepper.encodeDf('OHE', prepper.df_test)
    num_numeric = prepper.numerical_indices_np_end
    mean_X, std_X = (
        np.concatenate((np.mean(train_X[:, :num_numeric], axis=0), np.zeros(train_X.shape[1] - num_numeric)), axis=0),
        np.concatenate((np.std(train_X[:, :num_numeric], axis=0), np.ones(train_X.shape[1] - num_numeric)), axis=0)
        )
    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X      # standardize the train data
    # X = torch.tensor(X)

    X_test = (test_X - mean_X) / std_X  # standardize the test data
    # X_test = torch.tensor(X_test, dtype=torch.float32)
   

    test_X_ori_fmt = np.concatenate(
        (prepper.df_test.iloc[:, prepper.info['num_col_idx']],
        prepper.df_test.iloc[:, prepper.info['cat_col_idx']]), 
        axis=1)
    test_X_ordinal_fmt = prepper.encodeDf('Ordinal', prepper.df_test)
    orig_mask = generate_mask(test_X_ordinal_fmt, mask_type=mask_type, mask_num=num_trials, p=ratio)
                

    test_masks = prepper.extend_mask(orig_mask, encoding='OHE') # missing data in OHE format
    # # mask_tests = torch.tensor(test_masks)
    print("\n",test_X_ordinal_fmt, orig_mask, test_masks)

    # models_dir = f'saved_models/{args.dataname}/'
    # # model_path = os.path.join(models_dir, f"hyperimpute_{plugin}.pt")
    
    


    plugins = [
        # "hyperimpute",
        'miracle',
        'gain'
        ]
    
    for plugin in plugins:
        MSEs, ACCs = [], []
        rec_Xs = []
        print(f"Using plugin: {plugin}")
        model = Imputers().get(plugin, random_state=42) # use Imputers to get the model
        # model = Imputers().get(plugin, optimizer="hyperband", classifier_seed=["logistic_regression"], regression_seed=["linear_regression"])
        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            # Set up the HyperImpute imputer, define the model and training parameters
            x, x_miss, mask = ampute(test_X_ordinal_fmt, ampute_mechanism, p_miss)
            # print(X, x, x_miss, mask)
           
            X_pred_dec = model.fit_transform(x_miss.copy()).astype(float) # fit_transform the model on the missing data
            # X_pred_dec = model.fit_transform(x_miss.copy())
            loss = RMSE(X_pred_dec.values, x.values, mask.values)
            print(f"{X_pred_dec}, Plugin: {plugin}: Loss: {loss}")

            mse = masked_mse1(X_pred_dec, x, mask, mask_marks_missing=True)

            # print("mse =", mse)  
            # X_true = X_test.numpy()
            # X_true_dec = prepper.decodeNp(scheme='OHE', arr=X_true)
            # X_pred_dec = prepper.decodeNp(scheme='OHE', arr=X_pred)
            # mse, acc = get_eval(X_pred_dec, X_true_dec, orig_mask[trial], num_numeric)
            # mse = get_eval(X_pred_dec, test_X_ordinal_fmt, mask)
            print(f"Trial {trial + 1}/{num_trials} completed. MSE: {mse}")
            MSEs.append(mse)

        MSEs = np.array(MSEs)    
        print(MSEs)
        experiment_path = f'experiments/hyperimpute_imputation.csv'
        directory = os.path.dirname(experiment_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(experiment_path):
            columns = [
                "Dataset",
                "Method",
                "Mask Type",
                "Ratio",
                "Avg MSE",
                "STD of MSE",
                "Avg Acc",
                "STD of Acc"
            ]
            exp_df = pd.DataFrame(columns=columns)
        else:
            exp_df = pd.read_csv(experiment_path).drop(columns=['Unnamed: 0'])

        new_row = {"Dataset": dataname,
                "Method": f"HyperImpute_{plugin}",
                "Mask Type": args.mask,
                "Ratio": ratio,
                "Avg MSE": np.mean(MSEs),
                "STD of MSE": np.std(MSEs),
                # "Avg Acc": np.mean(ACCs),
                # "STD of Acc": np.std(ACCs)
                "Avg Acc": '',
                "STD of Acc": ''
                }
        new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
        new_df.to_csv(experiment_path)


if __name__ == '__main__':
    
    
    main()