import os
import json
import numpy.random
import torch
from be_great import GReaT
import warnings
import numpy as np
import logging
import argparse
import pandas as pd
import time
from tqdm import tqdm
from generate_mask import generate_mask
from dataset import Preprocessor, get_eval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("transformers.trainer").setLevel(logging.INFO)

# Suppress one-off warnings from tokenizer/config/etc.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

# PyTorch tracing helper warning
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--mask', type=str, default='MAR', help='Masking mechanisms.')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of sampling times.')
    parser.add_argument('--ratio', type=str, default="0.25", help='Masking ratio.')
    torch.manual_seed(42)
    numpy.random.seed(42)
    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    dataname = args.dataname
    device = args.device
    mask_type = args.mask
    ratio = float(args.ratio)
    num_trials = args.num_trials
    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    infopath = f'datasets/Info/{args.dataname}.json'
    info = None
    with open(infopath, 'r') as f:
        info = json.load(f)

    prepper = Preprocessor(dataname)
    train_df = pd.read_csv(f'datasets/{args.dataname}/train.csv')
    test_df = pd.read_csv(f'datasets/{args.dataname}/test.csv')
    num_cols = info['num_col_idx']
    cat_cols = info['cat_col_idx']
    num_numeric = len(num_cols)
    train_df = train_df.iloc[:, num_cols + cat_cols]
    test_df = test_df.iloc[:, num_cols + cat_cols]
    mean_X = train_df.iloc[:, :num_numeric].mean()
    std_X = train_df.iloc[:, :num_numeric].std()

    standardized_test_num = (test_df.iloc[:, :num_numeric] - mean_X)/std_X
    test_eval = np.concatenate((standardized_test_num.values, test_df.iloc[:, num_numeric:].values), axis=1)

    # Code for generating masks: Must be consistent across all baselines
    test_X_ori_fmt = np.concatenate((prepper.df_test.iloc[:, prepper.info['num_col_idx']],
                                     prepper.df_test.iloc[:, prepper.info['cat_col_idx']]), axis=1)
    test_X_ordinal_fmt = prepper.encodeDf('Ordinal', prepper.df_test)
    orig_mask = generate_mask(test_X_ordinal_fmt[:], mask_type=mask_type, mask_num=num_trials, p=ratio)

    MSEs, ACCs = [], []
    models_dir = f'saved_models/{args.dataname}/GReaT'
    great_model = GReaT.load_from_dir(models_dir)
    # max_length = 200 if args.dataname not in ['bean', 'default', 'gesture'] else 350  # some datasets need more tokens
    cat_edge_case = True if args.dataname == 'adult' else False
    with torch.no_grad():
        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            mask = orig_mask[trial]
            masked_df = test_df.mask(mask)
            imputed_data = great_model.impute(masked_df)
            standardized_imputed_num = (imputed_data.iloc[:, :num_numeric] - mean_X)/std_X
            standardized_imputed_num = standardized_imputed_num.fillna(0.0)  # for numerical features that were not generated within the max retries
            imputed_eval = np.concatenate((standardized_imputed_num, imputed_data.iloc[:, num_numeric:]), axis=1)
            mse, acc = get_eval(imputed_eval, test_eval, mask, num_numeric, cat_edge_case)
            MSEs.append(mse)
            ACCs.append(acc)

    MSEs = np.array(MSEs)
    ACCs = np.array(ACCs)
    experiment_path = f'experiments/imputation.csv'
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
               "Method": "GReaT",
               "Mask Type": args.mask,
               "Ratio": ratio,
               "Avg MSE": np.mean(MSEs),
               "STD of MSE": np.std(MSEs),
               "Avg Acc": np.mean(ACCs),
               "STD of Acc": np.std(ACCs)
               }
    new_df = pd.concat([exp_df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(experiment_path)



