# train_hyperimpute_min.py
import os, pandas as pd, pickle
import numpy as np
from tqdm import tqdm
from hyperimpute.plugins.imputers import Imputers
from dataset import Preprocessor, get_eval
from generate_mask import generate_mask
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from hyperimpute.utils.serialization import load, save


# --- stability on macOS / BLAS ---
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS",
          "CATBOOST_THREAD_COUNT","XGB_NUM_THREADS"]:
    os.environ[k] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

# 2) Generate mask based on the amputation mechanism
def ampute(x, mechanism, p_miss):
    x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)


# 1) Load your data (NaNs allowed)
dataname = "adult"  # or any other dataset

models_dir = f'saved_models/{dataname}/'

train = pd.read_csv(f"./datasets/{dataname}/train.csv")     # source train data file
test  = pd.read_csv(f"./datasets/{dataname}/test.csv")      # source test data file

# prepare the data: train and test data (true), test data (with missing values), and the mask
prepper = Preprocessor(dataname)
train_X = prepper.encodeDf('OHE', prepper.df_train)     # train_X is a numpy array
test_X = prepper.encodeDf('OHE', prepper.df_test)       #   test_X is a numpy array
num_numeric = prepper.numerical_indices_np_end          # index of the last numeric column

# print("train_X:", train_X)
# print("test_X:", test_X)        
# print("num_numeric:", num_numeric)

mean_X, std_X = (
    np.concatenate((np.mean(train_X[:, :num_numeric], axis=0), np.zeros(train_X.shape[1] - num_numeric)), axis=0),
    np.concatenate((np.std(train_X[:, :num_numeric], axis=0), np.ones(train_X.shape[1] - num_numeric)), axis=0)
    )
# print("mean_X:", mean_X, "std_X:", std_X)

# exit()
in_dim = train_X.shape[1]
X = (train_X - mean_X) / std_X
# X = torch.tensor(X)
X_test = (test_X - mean_X) / std_X
# X_test = torch.tensor(X_test, dtype=torch.float32)

test_X_ori_fmt = np.concatenate((prepper.df_test.iloc[:, prepper.info['num_col_idx']],
                                    prepper.df_test.iloc[:, prepper.info['cat_col_idx']]), axis=1)
test_X_ordinal_fmt = pd.DataFrame(prepper.encodeDf('Ordinal', prepper.df_test))

# Keep original index if you have a DF you transformed
idx = test_X_ordinal_fmt.index if isinstance(test_X_ordinal_fmt, pd.DataFrame) else None

df_out = pd.DataFrame(test_X_ordinal_fmt, index=idx)
out_path = os.path.join(models_dir, "test_X_ordinal_fmt.csv")
df_out.to_csv(out_path, index=False)

# print("Train X:", X)
# print("test_X_ori_fmt:", test_X_ori_fmt)
# print("test_X_ordinal_fmt:", test_X_ordinal_fmt)    

# exit()


mask_type = 'MCAR'  # or 'MAR', 'MCAR', 'MNAR_logistic_T2'
ratio = 0.25  # percentage of missing data
num_trials = 5  # number of trials for out-of-sample imputation


orig_mask = generate_mask(test_X_ordinal_fmt, mask_type=mask_type, mask_num=num_trials, p=ratio)
test_masks = prepper.extend_mask(orig_mask, encoding='OHE')

# print("Original Mask:", orig_mask.shape, orig_mask[:5])
# print("Test Masks:", test_masks.shape, test_masks[:5])

# exit()

# 2) Build the imputer
plugins = [
    "hyperimpute",  # or 
    "miracle",
    "gain"
    # "mean", "median", "knn", "mice", "missforest", "softimpute"
]

for plugin in plugins:
    print(f"Plugin: {plugin}")
    if plugin == "hyperimpute":
        imputer = Imputers().get(
            plugin,
            optimizer="hyperband",          # "simple" is faster, "bayesian" more thorough
            # n_jobs=1,                       # keep single-threaded for stability
            classifier_seed=["logistic_regression","random_forest","xgboost","catboost"],
            regression_seed=["linear_regression","random_forest_regressor","xgboost_regressor","catboost_regressor"],
            class_threshold=5,              # ≤5 uniques → treat as categorical
            imputation_order=2,             # random order across columns
            n_inner_iter=10,
            select_model_by_column=True,
            select_model_by_iteration=True,
            select_patience=5,
        )
    else: 
        imputer = Imputers().get(plugin, random_state=42)  # use Imputers to get the model
    # 3) Fit on training data (learn how to fill)
    imputer.fit(X.copy())
    # Save to a file

    buff = save(imputer)  # get the model as bytes

    # exit()
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"hyperimpute_{plugin}_final.pkl")

    # after imputer.fit(...)
    with open(path, "wb") as f:
        f.write(buff)   # .model() returns bytes
    # with open(f"{models_dir}/hyperimpute_{plugin}_final.pkl", "wb") as f:
    #     f.write(imputer.save())  # imputer.save() returns a bytes object
    print(f"Saved imputer model: {f.name}")

    # exit()  # exit after saving the model

    # 4) Impute
    test_x_true, test_x_missing, test_x_mask = ampute(test_X_ordinal_fmt, mask_type, ratio)  # simulate missing data
    # train_imp = imputer.transform(train)
    test_imp  = imputer.transform(test_x_missing.copy())  # impute the missing data

    # 5) Save artifacts to check the results later
    # train_imp.to_csv("train_x_imputed.csv", index=False)
    test_imp.to_csv(f"{models_dir}test_x_imputed_{plugin}.csv",  index=False)
    print("test imputed:", pd.DataFrame(test_imp))  

    mse = root_mean_squared_error(test_x_true.values, test_imp.values)  # RMSE
    rmse = RMSE(test_imp.values, test_x_true.values, test_x_mask.values)
    print(f"Plugin: {plugin}, RMSE: {rmse}, {mse*mse}")

    # with open(f"{models_dir}/hyperimpute_model_{plugin}.pkl","wb") as f:
    #     print(f"Saving imputer model: {f.name}")
    #     pickle.dump(imputer, f)
    

    
# print("test_imp:", test_imp.shape, test_imp[:5])
# print("test_X_ordinal_fmt:", test_X_ordinal_fmt.shape, pd.DataFrame(test_X_ordinal_fmt[:5]))
# print("test_mask:", test_masks.shape, pd.DataFrame(test_masks[:5]))
# mse, acc = masked_mse(test_imp, test_X_ordinal_fmt, test_masks, mask_marks_missing=True)

# exit()

# MSEs.append(mse)
# ACCs.append(acc)
# MSEs = []
# ACCs = []
# rec_Xs = []
# for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            # mask_test = mask_tests[trial]
            # mask_float = mask_test.float().to(device)
            # # x_t = torch.randn_like(X_test).to(device)
            # for t in range(args.timesteps - 1, -1, -1):
            #     timesteps = torch.full(size=(x_t.shape[0],), fill_value=t).to(device)
            #     alpha_t = diffusion_config['Alpha'][t].to(device)
            #     alpha_bar_t = diffusion_config['Alpha_bar'][t].to(device)
            #     alpha_bar_t_1 = diffusion_config['Alpha_bar'][t - 1].to(device) if t >= 1 else torch.tensor(1).to(
            #         device)
            #     sigma_t = diffusion_config['Sigma'][t].to(device)
            #     sigmas_predicted = model(x_t, timesteps)
            #     x_t = (x_t / torch.sqrt(alpha_t)) - (
            #             (1 - alpha_t) / (torch.sqrt(alpha_t) * torch.sqrt(1 - alpha_bar_t))) * sigmas_predicted

            #     x_t += diffusion_config['Sigma'][t] * torch.randn_like(x_t)

            #     x_cond_t = torch.sqrt(alpha_bar_t_1) * X_test_gpu + torch.sqrt(1-alpha_bar_t_1) * torch.randn_like(X_test_gpu)
            #     x_t = (1-mask_float) * x_cond_t + mask_float * x_t
            # X_pred = (x_t * mask_float + (1-mask_float) * X_test_gpu).cpu().numpy()
            # X_true = X_test.numpy()
            
# exit()
    # with open(f"{models_dir}/hyperimpute_model_{plugin}.pkl","wb") as f:
    #     pickle.dump(imputer, f)
    


    print(f"Saved: test_x_original.csv, test_x_imputed.csv, hyperimpute_model_{plugin}.pkl")
