import json

def build_latex():
    with open("results/dagnese_sota_proof_summary.json", "r") as f:
        data = json.load(f)
        
    print("% --- LATEX TABLE GENERATED FOR PAPER ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{\\textbf{DIF-FNO Topological Integrity and Sobolev Error Metrics across Irregular Domains.}}")
    print("\\label{tab:dagnese_sota_validation}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("\\textbf{Domain} & \\textbf{Rel. $L^2$ Error} & \\textbf{Sobolev $H^1$ Error} & \\textbf{Min $\\det(J_\\phi)$} & \\textbf{Grid Folding} \\\\")
    print("\\hline")
    
    for domain, metrics in data.items():
        l2 = metrics["relative_l2_error"]
        h1 = metrics["sobolev_h1_error"]
        det = metrics["min_det_jacobian"]
        folding = metrics["grid_folding_rate"]
        print(f"{domain} & {l2:.4f} & {h1:.4f} & {det:.6f} & \\textbf{{{folding}}} \\\\")
        
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    build_latex()
