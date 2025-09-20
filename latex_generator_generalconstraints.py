import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('experiments/general_constraints.csv').drop(columns=['Unnamed: 0'])
    datasets = df["Dataset"].unique()
    df['Method'] = df['Method'].replace(
        {'DiffPuter_Remastered': 'DiffPuter', 'harpoon_ohe': 'Harpoon'})
    df['Constraint'] = df['Constraint'].replace(
        {'range': 'Range', 'both': 'Both', 'category': 'Category'})
    # Start LaTeX table
    latex = ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{ll" + "cc" * len(datasets) + "}", "\\toprule"]

    # Header row
    header = ["Constraint", "Method"]
    for d in datasets:
        header.append("\\multicolumn{2}{c}{" + d + "}")
    latex.append(" & ".join(header) + " \\\\")
    subheader = ["", ""]
    for _ in datasets:
        subheader += ["Violation \%", "Alpha precision"]
    latex.append(" & ".join(subheader) + " \\\\")
    latex.append("\\midrule")

    constraint_order = {"Range": 0, "Category": 1, "Both": 2}
    df["ConstraintOrder"] = df["Constraint"].map(constraint_order)
    # Group by constraint then method
    for constraint, g1 in df.sort_values("ConstraintOrder").groupby("Constraint", sort=False):
        first_constraint = True
        for method, g2 in g1.groupby("Method"):
            row = []
            if first_constraint:
                row.append("\\multirow{" + str(len(g1["Method"].unique())) + "}{*}{" + constraint + "}")
                first_constraint = False
            else:
                row.append("")
            row.append(method)
            for d in datasets:
                sub = g2[g2["Dataset"] == d]
                if sub.empty:
                    row += ["--", "--"]
                else:
                    v = sub.iloc[0]
                    row.append(f"{v['Avg ViolationAcc']:.2f} $\\pm$ {v['Std ViolationAcc']:.2f}")
                    row.append(f"{v['Avg Alpha-P']:.2f} $\\pm$ {v['Std Alpha-P']:.2f}")
            latex.append(" & ".join(row) + " \\\\")
        latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    # Print final LaTeX
    print("\n".join(latex))
