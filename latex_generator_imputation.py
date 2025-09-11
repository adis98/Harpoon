import pandas as pd
import numpy as np
import argparse


def format_for_latex(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        if abs(x) >= 100:
            digits = np.log10(x)
            return f">$10^{int(digits)}$"
        else:
            return f"{x:.2f}"
    return x


def pivot_mask_acc(mask_type):
    method_order = [
        "GAIN",
        "Miracle",
        "GReaT",
        "Hyperimpute",
        "Remasker",
        "DiffPuter",
        "Harpoon-MSE",
        "Harpoon"
    ]
    df_mask = df[df['Mask Type'] == mask_type].copy()
    df_mask = df_mask[df_mask['Dataset'].isin(['adult', 'default', 'shoppers'])]
    df_mask['Avg Acc'] = df_mask['Avg Acc'].apply(format_for_latex)
    df_mask['Method'] = pd.Categorical(df_mask['Method'], categories=method_order, ordered=True)
    pivot = df_mask.pivot_table(
        index=['Ratio', 'Method'],
        columns='Dataset',
        values='Avg Acc',
        aggfunc='first'
    )
    pivot = pivot.sort_index(level=['Ratio', 'Method'])
    return pivot


def pivot_mask(mask_type):
    method_order = [
        "GAIN",
        "Miracle",
        "GReaT",
        "Hyperimpute",
        "Remasker",
        "DiffPuter",
        "Harpoon-MSE",
        "Harpoon"
    ]
    df_mask = df[df['Mask Type'] == mask_type].copy()
    df_mask['Avg MSE'] = df_mask['Avg MSE'].apply(format_for_latex)
    df_mask['Method'] = pd.Categorical(df_mask['Method'], categories=method_order, ordered=True)
    pivot = df_mask.pivot_table(
        index=['Ratio', 'Method'],
        columns='Dataset',
        values='Avg MSE',
        aggfunc='first'
    )
    pivot = pivot.sort_index(level=['Ratio', 'Method'])
    return pivot


# LaTeX generation with multirow for method
def generate_latex_multirow(pivot, caption, label):
    latex_rows = []
    for ratio, group in pivot.groupby(level=0):
        n_rows = len(group)
        first = True
        group_num = group.apply(pd.to_numeric, errors='coerce').fillna(np.inf)
        if 'acc' in label:
            best_indices = group_num.apply(lambda col: col.nlargest(2).index.tolist())
        else:
            best_indices = group_num.apply(lambda col: col.nsmallest(2).index.tolist())
        best_mask = pd.DataFrame(False, index=group_num.index, columns=group_num.columns)
        secondbest_mask = pd.DataFrame(False, index=group_num.index, columns=group_num.columns)
        for column in best_mask.columns:
            best_loc = best_indices.loc[0, column]
            secondbest_loc = best_indices.loc[1, column]
            best_mask.loc[best_loc, column] = True
            secondbest_mask.loc[secondbest_loc, column] = True
        group = group.where(~best_mask, "\\textbf{" + group.astype(str) + "}")
        group = group.where(~secondbest_mask, "\\underline{" + group.astype(str) + "}")
        for idx, row in group.iterrows():
            if first:
                ratio_cell = f"\\multirow{{{n_rows}}}{{*}}{{{ratio}}}"
                first = False
            else:
                ratio_cell = ""

            row_values = " & ".join(row.values)
            latex_rows.append(f"{ratio_cell} & {idx[1]} & {row_values} \\\\")
    latex_body = "\n".join(latex_rows)

    # Column headers: Method | Ratio | Dataset1 | Dataset2 | ...
    columns = ["Ratio", "Method"] + list(pivot.columns)
    col_str = " & ".join(columns) + " \\\\"

    latex_code = f"""
\\begin{{table}}[ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' * (len(columns))}}}
\\hline
{col_str}
\\hline
{latex_body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return latex_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Latex generator arguments')
    parser.add_argument('--mask', type=str, required=True, help='MAR, MCAR or MNAR scenario')
    parser.add_argument('--task', type=str, required=True, help='acc or mse table')
    args = parser.parse_args()
    df = pd.read_csv("experiments/imputation.csv").drop(columns=['Unnamed: 0'])
    # Only keep relevant columns
    df = df[['Dataset', 'Method', 'Mask Type', 'Ratio', 'Avg MSE', 'STD of MSE', 'Avg Acc', 'STD of Acc']]
    df = df[df['Dataset'] != 'news']
    df = df[df['Method'] != 'DiffPuter']
    df = df[df['Method'] != 'Hyperimpute']

    df = df[df['Ratio'].isin([0.25, 0.5, 0.75])]
    df['Ratio'] = df['Ratio'].map(lambda x: f"{x: .2f}")

    df['Method'] = df['Method'].replace({'DiffPuter_Remastered': 'DiffPuter', 'harpoon_ohe_mae': 'Harpoon', 'harpoon_ohe_mse': 'Harpoon-MSE'})
    df = df[df['Method'] != 'Harpoon-MSE']

    # Function to create pivot table per mask type
    if args.task == 'mse':
        table = pivot_mask(args.mask)
        latex = generate_latex_multirow(table, f'Imputation MSE for {args.mask} mask', f'tab:{args.mask}{args.task}')
    else:
        table = pivot_mask_acc(args.mask)
        latex = generate_latex_multirow(table, f'Imputation Acc. for {args.mask} mask', f'tab:{args.mask}{args.task}')

    print(latex)
