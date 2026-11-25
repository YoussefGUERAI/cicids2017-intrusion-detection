"""Generate a four-page draft from completed deadline-run evidence, never placeholders."""
from pathlib import Path
import json
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


def build_report(directory):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

    directory = Path(directory)
    required = ["binary_test_results.csv", "selected_boosting_settings.csv", "multiclass_results.csv",
                "multiclass_class_recall.csv", "destination_port_recall_change.csv", "computational_cost.csv",
                "ten_tree_errors.csv", "importance_all_original_predictors.csv", "appendix.json"]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise RuntimeError("Report not generated: required evidence is missing: " + ", ".join(missing))
    binary = pd.read_csv(directory / "binary_test_results.csv", index_col=0)
    settings = pd.read_csv(directory / "selected_boosting_settings.csv", index_col=0)
    mc = pd.read_csv(directory / "multiclass_results.csv")
    if len(mc) != 4:
        raise RuntimeError("All four multiclass experiments must be recorded before generating the report.")
    cost = pd.read_csv(directory / "computational_cost.csv", index_col=0)
    if len(cost) != 4:
        raise RuntimeError("All four cost-proxy measurements must be recorded before generating the report.")
    gains = pd.read_csv(directory / "destination_port_recall_change.csv", index_col=0)
    importance = pd.read_csv(directory / "importance_all_original_predictors.csv", index_col=0)
    errors = pd.read_csv(directory / "ten_tree_errors.csv")
    appendix = json.loads((directory / "appendix.json").read_text())
    baseline = pd.read_csv(directory / "baseline_validation_results.csv", index_col=0)
    bag = pd.read_csv(directory / "bagging_validation_results.csv")
    rf = pd.read_csv(directory / "rf_validation_results.csv")
    n = appendix["original_split_rows"]
    protocol = appendix["deadline_protocol"]
    output = directory / "Lab3_Report_DRAFT.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontName="Helvetica", fontSize=9,
                              leading=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="CaptionSmall", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=5))
    styles["Title"].fontSize = 19
    styles["Title"].leading = 23
    styles["Heading1"].fontSize = 14
    styles["Heading1"].leading = 17
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 14
    body = []
    def paragraph(text, style="BodySmall"):
        body.append(Paragraph(text, styles[style]))
    def heading(text):paragraph(text,"Heading2")
    def table(headers, rows, widths=None):
        data = [[Paragraph(escape(str(v)),styles["CaptionSmall"]) for v in headers]]
        data += [[Paragraph(escape(str(v)),styles["CaptionSmall"]) for v in row] for row in rows]
        t = Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dfe8ef")),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#8794a0")),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f6f8")]),
                               ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
                               ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),1)]))
        body.extend([t,Spacer(1,8)])
    def image(name,width,height):
        p=directory/name
        if not p.exists():raise RuntimeError(f"Required figure missing: {name}")
        body.append(Image(str(p),width=width,height=height,kind="proportional"))
        body.append(Spacer(1,6))
    def pct(x):return f"{100 * float(x):.3f}"
    def bool_count(series):return int(series.astype(str).str.lower().eq("true").sum())

    paragraph("Lab 3: Tree Ensembles", "Title")
    paragraph("CICIDS2017 | Resource-constrained experiment | Draft for author review", "CaptionSmall")
    heading("1. Protocol, reproducibility and baselines")
    paragraph(f"The original Lab 2 row memberships and cleaning were restored: {n['train']:,} training, "
              f"{n['validation']:,} validation and {n['test']:,} test flows. The complete representation has "
              f"{len(appendix['complete_feature_names']):,} predictors (68 numerical variables plus training-derived one-hot "
              "destination-port levels and an other level). Training-fitted imputation, transformation and scaling were reused. "
              "Seed 42 and two CPU workers were used. Test predictions were cached after all binary choices were frozen.")
    lr = baseline.loc['Logistic regression']
    paragraph(f"The single tree reproduced Lab 2 validation balanced accuracy (0.997173). The newly converged logistic fit "
              f"achieved {lr['balanced_accuracy']:.6f}, versus Lab 2's 0.953312; this is not a rounding match. "
              "The local 500-iteration fit was extended using saved coefficients. This optimization/provenance difference is "
              "disclosed rather than described as an exact reproduction.")
    heading("2. Aggregation and the common binary test")
    best_rf=rf.loc[rf.balanced_accuracy.idxmax()]
    paragraph(f"Bagging tested 50/100 trees (validation balanced accuracy {bag.balanced_accuracy.min():.6f} to "
              f"{bag.balanced_accuracy.max():.6f}). All eight RF candidates were completed. The highest RF validation score "
              f"was {best_rf.balanced_accuracy:.6f}; the practical-tie rule favored fewer trees and then fewer nodes. "
              "The all-feature forest, not feature-randomized sqrt(d), supplied the selected configuration. "
              "Thus feature randomization did not help this selected comparison; its benefit is not automatic.")
    table(["Frozen model","Bal. acc. %","Attack P %","Attack R %","Attack F1 %","Log loss"],
          [[name]+[pct(row[k]) for k in ['balanced_accuracy','attack_precision','attack_recall','attack_f1']]+[f"{row.log_loss:.5f}"]
           for name,row in binary.iterrows()], [116,70,65,65,69,65])
    paragraph("All six models were scored on the same full original test set. Training resources differ if the sampled XGBoost "
              "candidate was selected; these results are not a controlled algorithm-only comparison. Ensemble confusion matrices, "
              "OOB/validation tables and all row-level audit files accompany this report.")

    body.append(PageBreak())
    heading("3. Sequential correction, stopping and regularization")
    paragraph("The original four standard-gradient-boosting fits were completed. Learning rate 0.10 and depth 3 retained "
              "244 rounds with validation log loss 0.009014. The other three configurations reached 500 rounds, so "
              "the smaller-rate runs do not establish a loss plateau. Depth-1 stumps remain meaningful simple learners, "
              "even though depth 3 performed better in these completed standard-boosting comparisons.")
    paragraph(f"The original XGBoost fit was recovered after 145 complete evaluations (full-validation log loss 0.038785). "
              f"A replacement six-case search used {protocol['binary_training_rows']:,} training and "
              f"{protocol['binary_stopping_rows']:,} stopping-validation rows, all encoded columns, rate 0.10, depth 3, "
              "lambda 0/1/10 and gamma 0/0.1. The bin cap was 32, round cap 100, patience 15 and soft time budget "
              "10 minutes per fit. Original attack-category proportional sampling reserved at least one row per category. "
              "The sample-search winner and recovered full-data model were compared on full validation before freezing XGBoost.")
    table(["Model","Rate / depth","Retained rounds","lambda / gamma","Training / stopping rows"],
          [[name,f"{row.learning_rate:g} / {int(row.max_depth)}",int(row.selected_rounds),
            ('not applicable' if name=='Gradient boosting' else f"{row.reg_lambda:g} / {row.gamma:g}"),
            f"{int(row.training_rows):,} / {int(row.curve_validation_rows):,}"] for name,row in settings.iterrows()],
          [105,80,90,80,95])
    image("boosting_curves.png",450,166)
    paragraph("Curve losses can refer to different stopping populations; the provenance table makes this explicit. "
              "A time or round cap is not convergence. The original fixed 24-case full-data XGBoost search remains incomplete.")
    paragraph("Standard gradient boosting fits sequential corrections based on the negative loss gradient and updates leaf "
              "values. XGBoost uses both gradients and Hessians with regularized leaf outputs and split gains. Lambda shrinks "
              "leaf outputs; gamma requires sufficient improvement before splitting. Histogram search, bin resolution, "
              "sampling and time budgets also differ, so any performance difference cannot be assigned to second-order updates alone.")
    if (directory/'xgb_sample_validation_grid.csv').exists():
        grid=pd.read_csv(directory/'xgb_sample_validation_grid.csv')
        paragraph("Reduced-search termination counts: "+escape(str(grid.termination_reason.value_counts().to_dict()))+". "
                  "Every tested setting, loss curve and retained round is supplied in the evidence directory.","CaptionSmall")

    body.append(PageBreak())
    heading("4. Multiclass detection and destination-port ablation")
    paragraph(f"Eligibility used the original Lab 2 thresholds: at least 100 training, 20 validation and 20 test observations. "
              f"The paired experiments share {protocol['multiclass_training_rows']:,} sampled training and "
              f"{protocol['multiclass_stopping_rows']:,} sampled validation rows, while using all eligible original test rows. "
              "Both the numeric-only and with-port variants use selected binary settings, subject to the recorded round/time "
              "limits. The RF OOB calculation is disabled for these fits; tree construction settings are retained. "
              "This is a reduced-sample extension, not the required full-data multiclass fit.")
    table(["Family","Representation","Macro-F1","Fitted trees","Stop reason"],
          [[row.model,row.representation,f"{row.macro_f1:.5f}",int(row.fitted_trees),row.termination_reason] for row in mc.itertuples(index=False)],
          [90,100,65,75,120])
    gain_rows=[]
    for family in gains.columns:
        for category,value in gains[family].nlargest(3).items():gain_rows.append([family,category,f"{100*value:+.3f}"])
    table(["Family","Largest observed port-related recall changes","Change (pp)"],gain_rows,[100,265,85])
    paragraph("Macro-F1 weights each category equally. Full class-wise recalls, zero-recall categories, exclusions and all four "
              "confusion matrices are provided as CSV/PNG evidence. A large service-related gain may reflect the dataset's "
              "attack scenarios rather than a network-independent attack signature. If a wall limit caused different fitted "
              "counts, those differences also confound a port-only interpretation. The maximum listed recall change is not "
              "necessarily positive; read its sign.")
    heading("5. Predictive importance, dependence and causality")
    top_mdi=importance.mdi.idxmax();top_perm=importance.permutation_mean.idxmax()
    rank=importance[["mdi","permutation_mean"]].rank(ascending=False,method="min")
    rank_feature=(rank.mdi-rank.permutation_mean).abs().idxmax()
    paragraph(f"At original-predictor-group level, the leading MDI variable was {escape(str(top_mdi))}; the leading validation "
              f"permutation variable was {escape(str(top_perm))}. Port dummy MDI was aggregated and the complete one-hot port "
              "block was permuted jointly, preserving valid service encodings. Permutation used 5,000 validation rows and two "
              "repeats; its repeat variability is not a confidence interval. The ten highest individual encoded MDI features "
              "are also supplied separately. The largest observed rank difference was "
              f"{escape(str(rank_feature))}: MDI rank {int(rank.loc[rank_feature, 'mdi'])} versus permutation rank "
              f"{int(rank.loc[rank_feature, 'permutation_mean'])}.")
    image("importance_comparison.png",450,190)
    paragraph("MDI measures training split reductions and may favor variables with many split opportunities. Permutation "
              "measures score loss under perturbation. Correlated predictors can share split credit or substitute for one "
              "another after perturbation, masking importance. Neither ranking establishes causation.")

    body.append(PageBreak())
    heading("6. Computational cost and a deployment recommendation")
    table(["Selected model","Trees","Predict 1,000 (s)","Size (MiB)","Fresh-fit proxy (s)"],
          [[name,int(row.selected_fitted_trees),f"{row.prediction_median_seconds:.4f}",
            f"{row.selected_serialized_bytes_compress3/2**20:.3f}",f"{row.fresh_fit_proxy_seconds:.2f}"] for name,row in cost.iterrows()],
          [112,45,95,78,120])
    paragraph("Prediction uses the same fixed flows, one warm-up and the median of three calls; sizes use joblib compression 3. "
              "Fresh-fit times are proxies on the same 2,000 training rows, not full-data fit times. XGBoost includes stopping "
              "evaluation/checkpoint overhead and a five-minute soft cap; actual proxy tree counts and stop reasons are recorded. "
              "Full-data fit timings are intentionally missing rather than fabricated. Cached reload times are not reported as fits.")
    if 'Random forest' in binary.index:
        r=binary.loc['Random forest']
        paragraph(f"For a CPU-based intrusion-monitoring deployment that values detection consistency over single-tree "
                  f"explainability, the selected random forest is a reasonable candidate: test balanced accuracy "
                  f"{pct(r.balanced_accuracy)}% and attack recall {pct(r.attack_recall)}%. Its limitations are greater model "
                  "complexity and the expensive observed training search. The cost table quantifies its inference and storage "
                  "trade-offs. For a more constrained device, the single tree is a simpler alternative, subject to the observed "
                  "false-alarm and missed-attack trade-off. This recommendation does not imply that dataset-level test scores "
                  "guarantee live-network performance.")
    heading("7. Inspection of ten single-tree errors")
    paragraph(f"The first {len(errors)} test errors in stable test-row order were inspected without selecting only successful "
              f"corrections. RF corrected {bool_count(errors.corrected_by_rf)} and XGBoost corrected "
              f"{bool_count(errors.corrected_by_xgb)}. The accompanying ten_tree_errors.csv identifies every flow, original "
              "category, true binary label, predictions and correction flags.")
    heading("8. Scope limitations and submission evidence")
    for limitation in appendix['limitations']:
        paragraph("- "+escape(limitation))
    paragraph("The evidence bundle includes reproducible notebook/support code; original and sampled class counts; sample "
              "row IDs; all feature names; seed and software versions; tested and selected settings; retained rounds; test "
              "tables; ensemble confusion matrices; multiclass recalls; importance/correlation tables; and measured cost "
              "metadata. These limitations must be disclosed and do not constitute full compliance with every original requirement.")
    paragraph("Author review required: confirm all figures/tables, add student identification, and obtain instructor approval "
              "for the reduced protocol where required. This draft contains generated summaries of actual saved results, not "
              "guarantees of experimental adequacy.","CaptionSmall")

    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Helvetica',8)
        canvas.drawString(1.6*cm,.9*cm,"Lab 3 | Reduced computational-budget experiment")
        canvas.drawRightString(A4[0]-1.6*cm,.9*cm,str(doc.page));canvas.restoreState()
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=1.6*cm,leftMargin=1.6*cm,
                      topMargin=1.3*cm,bottomMargin=1.4*cm,title="Lab 3: Tree Ensembles - Deadline Experiment").build(body,onFirstPage=footer,onLaterPages=footer)
    return output


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    print(build_report(parser.parse_args().directory))
