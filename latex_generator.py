import pandas as pd
import numpy as np


def format_for_latex(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        if abs(x) >= 100:
            digits = np.log10(x)
            return f">$10^{int(digits)}$"
        else:
            return f"{x:.2f}"
    return x


def pivot_mask_acc(mask_type):
    df_mask = df[df['Mask Type'] == mask_type].copy()
    df_mask = df_mask[df_mask['Dataset'].isin(['adult', 'default', 'shoppers'])]
    df_mask['Avg Acc'] = df_mask['Avg Acc'].apply(format_for_latex)

    pivot = df_mask.pivot_table(
        index=['Method', 'Ratio'],
        columns='Dataset',
        values='Avg Acc',
        aggfunc='first'
    )
    return pivot


def pivot_mask(mask_type):
    df_mask = df[df['Mask Type'] == mask_type].copy()
    df_mask['Avg MSE'] = df_mask['Avg MSE'].apply(format_for_latex)

    pivot = df_mask.pivot_table(
        index=['Method', 'Ratio'],
        columns='Dataset',
        values='Avg MSE',
        aggfunc='first'
    )
    return pivot


# LaTeX generation with multirow for method
def generate_latex_multirow(pivot, caption, label):
    latex_rows = []
    for method, group in pivot.groupby(level=0):
        n_rows = len(group)
        first = True
        for idx, row in group.iterrows():
            if first:
                method_cell = f"\\multirow{{{n_rows}}}{{*}}{{{method}}}"
                first = False
            else:
                method_cell = ""
            row_values = " & ".join(row.values)
            latex_rows.append(f"{method_cell} & {idx[1]} & {row_values} \\\\")
    latex_body = "\n".join(latex_rows)

    # Column headers: Method | Ratio | Dataset1 | Dataset2 | ...
    columns = ["Method", "Ratio"] + list(pivot.columns)
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
    df = pd.read_csv("experiments/imputation.csv").drop(columns=['Unnamed: 0'])

    # Only keep relevant columns
    df = df[['Dataset', 'Method', 'Mask Type', 'Ratio', 'Avg MSE', 'STD of MSE', 'Avg Acc', 'STD of Acc']]
    df = df[df['Dataset'] != 'news']
    df = df[df['Method'] != 'DiffPuter']
    df = df[df['Ratio'].isin([0.25, 0.75])]
    df['Ratio'] = df['Ratio'].map(lambda x: f"{x: .2f}")

    df['Method'] = df['Method'].replace({'DiffPut_Remastered': 'DiffPuter', 'harpoon_basicmanifold_ohe': 'Harpoon OHE', 'Harpoon_ordinal_basicmanifold': 'Harpoon ordinal'})

    # Function to create pivot table per mask type

    # Create tables for each mask type
    # mar_table = pivot_mask("MAR")
    # mcar_table = pivot_mask("MCAR")
    mnar_table = pivot_mask("MNAR")
    # mar_table_acc = pivot_mask_acc("MAR")
    # mcar_table_acc = pivot_mask_acc("MCAR")
    mnar_table_acc = pivot_mask_acc("MNAR")

    # Generate LaTeX
    # mar_latex = generate_latex_multirow(mar_table, "MAR Masked Data MSE", "tab:mar_mse")
    # mcar_latex = generate_latex_multirow(mcar_table, "MCAR Masked Data MSE", "tab:mcar_mse")
    mnar_latex = generate_latex_multirow(mnar_table, "MNAR Masked Data MSE", "tab:mnar_mse")

    # mar_latex_acc = generate_latex_multirow(mar_table_acc, "MAR Masked Data Acc", "tab:mar_acc")
    # mcar_latex_acc = generate_latex_multirow(mcar_table_acc, "MCAR Masked Data Acc", "tab:mcar_acc")
    mnar_latex_acc = generate_latex_multirow(mnar_table_acc, "MNAR Masked Data Acc", "tab:mnar_acc")

    # Print LaTeX code (can copy-paste into Overleaf)
    # print("=== MAR ===")
    # print(mar_latex)
    # print(mar_latex_acc)
    # print("\n=== MCAR ===")
    # print(mcar_latex)
    # print(mcar_latex_acc)
    # print("\n=== MNAR ===")
    print(mnar_latex)
    print(mnar_latex_acc)
