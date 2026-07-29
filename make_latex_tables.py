"""Emit the four appendix tables comparing SAEgis on Qwen2.5-VL-3B and Gemma-4-E2B.

Both models are evaluated by saegis_tables.py on identical images with identical code,
so the numbers are directly comparable -- the Qwen column is recomputed here rather than
copied from the paper (it reproduces the paper's Table 1 to mean |dF1| = 0.21).

Only the single-layer SAEgis variant appears: the Gemma SAE was trained at one location
(the vision->LM projection), so ensemble rows and the dense/PIP baselines are out of
scope.
"""

import argparse
import json

ATTACKS = ["SSA-CWA", "M-Attack", "FOA-Attack"]
MODELS = [("Qwen2.5-VL-3B", "qwen"), ("Gemma-4-E2B", "gemma")]

PREAMBLE = r"""% Requires \usepackage{booktabs,multirow}
"""


def fmt(c):
    return f"{c['P']:.1f} & {c['R']:.0f} & {c['F1']:.1f}"


def attack_header(group_label):
    cols = " & ".join(rf"\multicolumn{{3}}{{c}}{{{a}}}" for a in ATTACKS)
    cmids = " ".join(rf"\cmidrule(lr){{{3+3*i}-{5+3*i}}}" for i in range(3))
    return (
        rf"\toprule" "\n"
        rf"& & {cols} \\" "\n"
        rf"{cmids}" "\n"
        rf"{group_label} & Model & P & R & F1 & P & R & F1 & P & R & F1 \\" "\n"
        rf"\midrule"
    )


def block(group, rows, res):
    """One \multirow group: the same setting evaluated for each model."""
    out = [rf"\multirow{{2}}{{*}}{{{group}}}"]
    for i, (disp, key) in enumerate(MODELS):
        cells = " & ".join(fmt(res[key][rows][a]) for a in ATTACKS)
        prefix = "" if i == 0 else ""
        out.append(rf"{prefix} & {disp} & {cells} \\")
    return "\n".join(out)


def table_1(res):
    body = []
    for i, d in enumerate(["NIPS17", "LLaVA", "Medical"]):
        rows = [rf"\multirow{{2}}{{*}}{{{d}}}"]
        for disp, key in MODELS:
            cells = " & ".join(fmt(res[key]["table1"][d][a]) for a in ATTACKS)
            rows.append(rf" & {disp} & {cells} \\")
        body.append("\n".join(rows))
    return (
        r"\begin{table}[t]" "\n" r"\centering" "\n"
        r"\caption{In-domain detection, SAEgis single-layer, Qwen2.5-VL-3B vs Gemma-4-E2B. "
        r"Features are selected on the same dataset and attack they are evaluated on "
        r"($K{=}256$), and the threshold is the 98th percentile of clean dev scores "
        r"(FPR${=}0.02$).}" "\n"
        r"\label{tab:appendix-in-domain}" "\n"
        r"\begin{tabular}{llccccccccc}" "\n"
        + attack_header("Data") + "\n"
        + "\n\\midrule\n".join(body) + "\n"
        r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}"
    )


def table_2(res):
    keys = list(res["qwen"]["table2"].keys())
    body = []
    for k in keys:
        label = k.replace("->", r" $\rightarrow$ ")
        rows = [rf"\multirow{{2}}{{*}}{{{label}}}"]
        for disp, mk in MODELS:
            cells = " & ".join(fmt(res[mk]["table2"][k][a]) for a in ATTACKS)
            rows.append(rf" & {disp} & {cells} \\")
        body.append("\n".join(rows))
    return (
        r"\begin{table}[t]" "\n" r"\centering" "\n"
        r"\caption{Cross-domain detection. Attack-relevant features are selected on the "
        r"source dataset and applied unchanged to the target dataset, keeping the attack "
        r"fixed.}" "\n"
        r"\label{tab:appendix-cross-domain}" "\n"
        r"\begin{tabular}{llccccccccc}" "\n"
        + attack_header("Transfer") + "\n"
        + "\n\\midrule\n".join(body) + "\n"
        r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}"
    )


def table_3(res):
    datasets = ["NIPS17", "LLaVA", "Medical"]
    header_cols = " & ".join(rf"\multicolumn{{3}}{{c}}{{{d}}}" for d in datasets)
    cmids = " ".join(rf"\cmidrule(lr){{{3+3*i}-{5+3*i}}}" for i in range(3))
    body = []
    for setting in ["SSA-CWA->M-Attack", "SSA-CWA->FOA-Attack"]:
        label = setting.replace("->", r" $\rightarrow$ ")
        rows = [rf"\multirow{{2}}{{*}}{{{label}}}"]
        for disp, mk in MODELS:
            cells = " & ".join(fmt(res[mk]["table3"][setting][d]) for d in datasets)
            rows.append(rf" & {disp} & {cells} \\")
        body.append("\n".join(rows))
    return (
        r"\begin{table}[t]" "\n" r"\centering" "\n"
        r"\caption{Cross-attack detection. Features are selected on SSA-CWA and evaluated "
        r"against an unseen attack on the same dataset.}" "\n"
        r"\label{tab:appendix-cross-attack}" "\n"
        r"\begin{tabular}{llccccccccc}" "\n"
        r"\toprule" "\n"
        rf"& & {header_cols} \\" "\n"
        f"{cmids}\n"
        r"Transfer Setting & Model & P & R & F1 & P & R & F1 & P & R & F1 \\" "\n"
        r"\midrule" "\n"
        + "\n\\midrule\n".join(body) + "\n"
        r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}"
    )


def table_4(res):
    rows = []
    for disp, mk in MODELS:
        t = res[mk]["table4"]
        i, c, d = t["in_domain"], t["cross_domain"], t["cross_attack_delta"]
        rows.append(
            rf"{disp} & {i['P']:.1f} & {i['R']:.1f} & {i['F1']:.1f} & "
            rf"{c['P']:.1f} & {c['R']:.1f} & {c['F1']:.1f} & "
            rf"{d['P']:+.1f} & {d['R']:+.1f} & {d['F1']:+.1f} \\"
        )
    return (
        r"\begin{table}[t]" "\n" r"\centering" "\n"
        r"\caption{Overall SAEgis results. In-domain and cross-domain columns average "
        r"their respective tables; the cross-attack columns are the change relative to "
        r"the corresponding no-transfer scores.}" "\n"
        r"\label{tab:appendix-overall}" "\n"
        r"\begin{tabular}{lccccccccc}" "\n"
        r"\toprule" "\n"
        r"& \multicolumn{3}{c}{In-domain} & \multicolumn{3}{c}{Cross-domain} & "
        r"\multicolumn{3}{c}{Cross-attack} \\" "\n"
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}" "\n"
        r"Model & P & R & F1 & P & R & F1 & $\Delta$P & $\Delta$R & $\Delta$F1 \\" "\n"
        r"\midrule" "\n" + "\n".join(rows) + "\n"
        r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="results_qwen.json")
    ap.add_argument("--gemma", default="results_gemma.json")
    ap.add_argument("--out", default="appendix_tables.tex")
    args = ap.parse_args()

    res = {"qwen": json.load(open(args.qwen)), "gemma": json.load(open(args.gemma))}
    tex = PREAMBLE + "\n\n".join([table_1(res), table_2(res), table_3(res), table_4(res)]) + "\n"
    with open(args.out, "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
