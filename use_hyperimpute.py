import os
import sys
import numpy as np
import argparse
import warnings
from tqdm import tqdm
from dataset import Preprocessor
from hyperimpute.plugins.imputers import Imputers

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Train HyperImpute on tabular datasets')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mask', type=str, default='MCAR', help='Masking mechanism: MCAR, MAR, MNAR_logistic_T2')
    parser.add_argument('--ratio', type=float, default=0.3, help='Missing ratio')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of mask trials')
    if any('ipykernel' in arg or 'jupyter' in arg for arg in sys.argv):
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()

def main():
    args = parse_args()
    dataname = args.dataname
    mask_type = args.mask
    ratio = args.ratio
    num_trials = args.num_trials

    print(f"Dataset: {dataname}, Mask: {mask_type}, Ratio: {ratio}, Trials: {num_trials}")

    # Load and encode dataset
    prepper = Preprocessor(dataname)
    X = prepper.encodeDf('OHE', prepper.df_train)  # One-hot encode categorical columns
    num_numeric = prepper.numerical_indices_np_end

    np.random.seed(42)
    masks = [(np.random.rand(*X.shape) < ratio) for _ in range(num_trials)]

    MSEs = []
    models_dir = f'saved_models/{dataname}/'
    os.makedirs(models_dir, exist_ok=True)

    # Set up the HyperImpute imputer
    imputer = Imputers().get(
        "hyperimpute",
        optimizer="hyperband",
        classifier_seed=[
            # "logistic_regression", 
            # "catboost", 
            # "xgboost", 
            "random_forest"],
        regression_seed=[
            # "linear_regression",
            # "catboost_regressor",
            # "xgboost_regressor",
            "random_forest_regressor",
        ],
        class_threshold=5,
        imputation_order=2,
        n_inner_iter=10,
        select_model_by_column=True,
        select_model_by_iteration=True,
        select_lazy=True,
        select_patience=5,
    )

    for trial in tqdm(range(num_trials), desc='HyperImpute Training'):
        X_miss = X.copy()
        X_miss[masks[trial]] = np.nan

        # Train imputer on masked data
        imputer.fit(X_miss)
        X_imputed = imputer.transform(X_miss)

        # Evaluate only on missing entries
        mse = np.nanmean((X_imputed[masks[trial]] - X[masks[trial]]) ** 2)
        MSEs.append(mse)

        # Save the imputer for each trial
        imputer.save(os.path.join(models_dir, f"hyperimpute_trial{trial}.pkl"))
        print(f"Trial {trial}: MSE={mse:.6f}")

    print(f"Avg MSE: {np.mean(MSEs):.6f} ± {np.std(MSEs):.6f}")

if __name__ == '__main__':
    main()