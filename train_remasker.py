import numpy as np
import pandas as pd
# from utils import get_args_parser
from remasker.remasker_impute import ReMasker
import os, pandas as pd, pickle
import numpy as np
import torch
from tqdm import tqdm
from hyperimpute.plugins.imputers import Imputers
from dataset import Preprocessor
from generate_mask import generate_mask
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from hyperimpute.utils.serialization import load, save
import argparse

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')


args = parser.parse_args()
# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'


# --- stability on macOS / BLAS ---
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS",
          "CATBOOST_THREAD_COUNT","XGB_NUM_THREADS"]:
    os.environ[k] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

# 2) Generate mask based on the amputation mechanism
# def ampute(x, mechanism, p_miss):
#     x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)
#
#     mask = x_simulated["mask"]
#     x_miss = x_simulated["X_incomp"]
#
#     return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # Initialize other args
    dataname = args.dataname
    device = args.device
    models_dir = f'saved_models/{dataname}/'

    train = pd.read_csv(f"datasets/{dataname}/train.csv")  # source train data file
    test = pd.read_csv(f"datasets/{dataname}/test.csv")  # source test data file

    # prepare the data: train and test data (true), test data (with missing values), and the mask
    prepper = Preprocessor(dataname)
    train_X = prepper.encodeDf('Ordinal', prepper.df_train)  # train_X is a numpy array
    test_X = prepper.encodeDf('Ordinal', prepper.df_test)  # test_X is a numpy array
    num_numeric = prepper.numerical_indices_np_end  # index of the last numeric column

    mean_X, std_X = (
        np.mean(train_X, axis=0), np.std(train_X, axis=0)
    )

    in_dim = train_X.shape[1]
    X = (train_X - mean_X) / std_X
    # X = torch.tensor(X)
    # X_test = (test_X - mean_X) / std_X

    mask_type = 'MCAR'  # or 'MAR', 'MCAR', 'MNAR_logistic_T2'
    ratio = 0.2  # train on MCAR 0.2

    train_mask = generate_mask(train_X, mask_type=mask_type, mask_num=1, p=ratio)[0]
    X_miss = X.copy()
    X_miss[train_mask] = np.nan

    imputer = ReMasker()
    remasker = imputer.fit(torch.as_tensor(X_miss, dtype=torch.float32))

    print(remasker.model)

    # Save to a file

    buff = save(imputer)  # get the model as bytes

    # exit()
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"remasker.pkl")

    # after imputer.fit(...)
    with open(path, "wb") as f:
        f.write(buff)  # .model() returns bytes
    # with open(f"{models_dir}/hyperimpute_{plugin}_final.pkl", "wb") as f:
    #     f.write(imputer.save())  # imputer.save() returns a bytes object
    print(f"Saved imputer model: {f.name}")

# test_X_ori_fmt = np.concatenate((prepper.df_test.iloc[:, prepper.info['num_col_idx']],
#                                     prepper.df_test.iloc[:, prepper.info['cat_col_idx']]), axis=1)
# test_X_ordinal_fmt = pd.DataFrame(prepper.encodeDf('Ordinal', prepper.df_test))
#
# # Keep original index if you have a DF you transformed
# idx = test_X_ordinal_fmt.index if isinstance(test_X_ordinal_fmt, pd.DataFrame) else None
#
# df_out = pd.DataFrame(test_X_ordinal_fmt, index=idx)
# out_path = os.path.join(models_dir, "test_X_ordinal_fmt.csv")
# df_out.to_csv(out_path, index=False)

# print("Train X:", X)
# print("test_X_ori_fmt:", test_X_ori_fmt)
# print("test_X_ordinal_fmt:", test_X_ordinal_fmt)    

# exit()

# mask_type = 'MCAR'  # or 'MAR', 'MCAR', 'MNAR_logistic_T2'
# ratio = 0.20  # percentage of missing data
# num_trials = 5  # number of trials for out-of-sample imputation
#
#
# orig_mask = generate_mask(test_X_ordinal_fmt, mask_type=mask_type, mask_num=num_trials, p=ratio)
# test_masks = prepper.extend_mask(orig_mask, encoding='OHE')
#
# print("Original Mask:", orig_mask.shape, orig_mask[:5])
# print("Test Masks:", test_masks.shape, test_masks[:5])

# exit()
# where you call imputer.fit(X)


# print(X)


# torch.save(remasker.model.state_dict(), self.path)
# imputed = imputer.fit_transform(X)


# exit()  # exit after saving the model

# 4) Impute
# test_x_true, test_x_missing, test_x_mask = ampute(test_X_ordinal_fmt, mask_type, ratio)  # simulate missing data
# train_imp = imputer.transform(train)
# test_imp  = remasker.transform(test_x_missing.copy())  # impute the missing data

# 5) Save artifacts to check the results later
# train_imp.to_csv("train_x_imputed.csv", index=False)
# test_imp.to_csv(f"{models_dir}test_x_imputed_remasker.csv",  index=False)
# print("test imputed:", pd.DataFrame(test_imp))

# mse = root_mean_squared_error(test_x_true.values, test_imp.values)  # RMSE
# rmse = RMSE(test_imp.values, test_x_true.values, test_x_mask.values)
# print(f"Remasker, RMSE: {rmse}, {mse*mse}")

# 3) Impute the missing values
# exit()