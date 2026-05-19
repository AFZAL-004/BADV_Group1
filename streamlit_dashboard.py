"""
Airline Route Profitability — Interactive Dashboard
COMM074 · Group Project

Run with: streamlit run streamlit_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import os
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Route Profitability",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
PROC      = BASE / "processed_data"
M_DIRS    = {
    "RF_DT_Gokulkrishna":  BASE / "RF_DT_Gokulkrishna",
    "XGBoost_DT_Joel":     BASE / "XGBoost_DT_Joel",
    "SVM_DT_Apprajit":     BASE / "SVM_DT_Apprajit",
    "LRPoly_DT_Afzal":     BASE / "LRPoly_DT_Afzal",
    "MLP_DT_Vino":         BASE / "MLP_DT_Vino",
}
CSV_MAP = {
    "RF_DT_Gokulkrishna": M_DIRS["RF_DT_Gokulkrishna"] / "rf_dt_gokulkrishna.csv",
    "XGBoost_DT_Joel":    M_DIRS["XGBoost_DT_Joel"]    / "xgb_dt_joel.csv",
    "SVM_DT_Apprajit":    M_DIRS["SVM_DT_Apprajit"]    / "svm_dt_apprajit.csv",
    "LRPoly_DT_Afzal":    M_DIRS["LRPoly_DT_Afzal"]    / "lrpoly_dt_afzal.csv",
    "MLP_DT_Vino":        M_DIRS["MLP_DT_Vino"]        / "mlp_dt_vino.csv",
}

MODEL_COLORS = {
    # Primary models
    "Random Forest"      : "#2E86AB",
    "XGBoost"            : "#F6AE2D",
    "SVM RBF"            : "#9B59B6",
    "LR+Poly"            : "#27AE60",
    "KNN"                : "#E67E22",
    "MLP"                : "#E74C3C",
    # Decision Trees — muted versions of each member's primary model colour
    "DT (Gokulkrishna)"  : "#4A6FA5",   # muted blue  (pairs with RF)
    "DT (Joel)"          : "#C9A227",   # muted gold  (pairs with XGBoost)
    "DT (Apprajit)"      : "#7D3C98",   # muted purple(pairs with SVM)
    "DT (Afzal)"         : "#1E8449",   # muted green (pairs with LR+Poly)
    "DT (Vino)"          : "#922B21",   # muted red   (pairs with MLP)
    # Generic fallback kept for any legacy references
    "Decision Tree"      : "#7F8C8D",
}

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    dfs = []
    for member, path in CSV_MAP.items():
        if path.exists():
            df = pd.read_csv(path)
            df["Member"] = member
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


@st.cache_data
def load_preprocessing_stats():
    path = PROC / "preprocessing_stats.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_names():
    path = PROC / "feature_names.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def bar_color(model_name):
    for key, color in MODEL_COLORS.items():
        if key.lower() in model_name.lower():
            return color
    return "#95A5A6"


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✈ Navigation")
    page = st.radio(
        "Go to",
        [
            "Project Overview",
            "Dataset Summary",
            "Model Comparison",
            "Member Deep-Dive",
            "Business Insights",
            "ML Prediction",
        ],
    )
    st.divider()
    st.caption("COMM074 · Airline Route Profitability")
    st.caption("Group Project · 2025–26")

all_results = load_results()
stats       = load_preprocessing_stats()
features    = load_feature_names()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Project Overview
# ═══════════════════════════════════════════════════════════════════════════
if page == "Project Overview":
    st.title("Airline Route Profitability Prediction")
    st.subheader("COMM074 — Group Project Overview")

    st.markdown("""
    This dashboard presents the results of a machine learning study predicting whether airline routes
    are **profitable** or **loss-making**, using operational and scheduling features observable before
    a route is operated.

    ---

    ### Business Problem

    Airlines invest significant capital in route planning. Incorrectly maintaining an unprofitable route
    costs an estimated **£42,000 per quarter** in excess operating costs. Incorrectly retiring a profitable
    route foregoes an estimated **£30,000 per quarter** in revenue.

    A reliable classification model enables data-driven route portfolio decisions, reducing both types of error.

    ---

    ### CRISP-DM Framework
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Business Understanding**\nRoute profitability prediction as binary classification. F1-Score primary metric.")
    with col2:
        st.info("**Data Understanding & Preprocessing**\n15,000 synthetic flight records, 37 engineered features, 80/10/10 split.")
    with col3:
        st.info("**Modelling & Evaluation**\n5 individual models + 5 independently tuned Decision Trees (different params per member). GridSearchCV/RandomizedSearchCV tuning.")

    st.divider()

    st.markdown("### Team Models")
    col1, col2, col3, col4, col5 = st.columns(5)
    model_info = [
        ("M1", "Random Forest", "#2E86AB", "Ensemble bagging. Native MDI importance."),
        ("M2", "XGBoost", "#F6AE2D", "Sequential boosting. Gain importance."),
        ("M3", "SVM (RBF)", "#9B59B6", "Kernel-based max margin. Permutation importance."),
        ("M4", "LR+Poly", "#27AE60", "Polynomial features (degree=2) + L2 logistic regression. Permutation importance."),
        ("M5", "MLP", "#E74C3C", "Neural network. Loss curve + permutation importance."),
    ]
    for col, (member, model, color, desc) in zip([col1, col2, col3, col4, col5], model_info):
        with col:
            st.markdown(
                f'<div style="background:{color}22;border-left:4px solid {color};padding:10px;border-radius:4px;">'
                f"<b>{member}</b><br><b>{model}</b><br><small>{desc}</small></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Decision Tree: Independently Tuned per Member")
    st.markdown("""
    Each member independently tunes a **Decision Tree** with a different hyperparameter search space, producing genuinely distinct models:
    - **M1 (RF)** — gini criterion, balanced class weights, shallow depth (max 7)
    - **M2 (XGBoost)** — entropy criterion, no class balancing, deeper growth (max 11)
    - **M3 (SVM)** — gini/entropy, balanced class weights, high min-samples regularisation
    - **M4 (LR+Poly)** — gini/entropy, no class balancing, deep/unconstrained growth
    - **M5 (MLP)** — entropy, heavy regularisation via large min_samples_leaf (≥ 15)

    The F1 delta between each member's primary model and their DT shows the value added by the more complex algorithm.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Dataset Summary
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Dataset Summary":
    st.title("Dataset Summary")
    st.subheader("Preprocessing & Feature Engineering")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Records", "15,000")
    col2.metric("Final Features", str(len(features)) if features else "37")
    col3.metric("Train / Val / Test", "11,997 / 1,500 / 1,500")
    col4.metric("Class Balance", "63.7% : 36.3%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Feature Engineering Highlights")
        st.markdown("""
        | Feature | Derivation |
        |---------|-----------|
        | `price_per_km` | Avg Ticket Price ÷ Flight Distance |
        | `delay_flag` | Delay_Minutes > 30 |
        | `is_weekend` | Extracted from departure date |
        | `is_narrow_body` | Aircraft_Type mapping |
        | `flight_month` | Month extracted from date |
        | `Season_Ordinal` | Off-Peak=0, Shoulder=1, Peak=2 |
        | `Demand_Level_Ordinal` | Low=0, Medium=1, High=2 |
        | `load_revenue_ratio` | Load_Factor × price_per_km |
        | `season_sin` | sin(2π × Season_Ordinal / 4) |
        | `season_cos` | cos(2π × Season_Ordinal / 4) |

        **Encoding:** One-hot for Route_Category, Alliance, Region (country→region mapping reduces cardinality)

        **Log1p transform:** Applied to right-skewed columns (Flight_Hours, Passengers, etc.)

        **IQR Winsorisation:** Flight_Hours, Competition_Index, Aircraft_Age_Years, Passengers
        """)

    with col2:
        st.markdown("### Data Leakage Prevention")
        st.markdown("""
        The following post-flight financial columns were **excluded** to prevent data leakage:
        - `Profit`, `Total_Revenue`, `Fuel_Cost`, `Maintenance_Cost`, `Staff_Cost`, `Airport_Fees`

        **Delay_Minutes / On_Time_Performance** are retained as route-level historical averages — observable at scheduling time from past operations data.

        **StandardScaler** fitted on training set only, applied to validation/test. Prevents test-set information leaking into scaling parameters.
        """)
        st.markdown("### Split Strategy")
        st.markdown("""
        **80 / 10 / 10 stratified split** preserves the 63.7/36.3 class distribution across all three sets.

        - Training: 11,997 rows — model fitting
        - Validation: 1,500 rows — hyperparameter selection
        - Test: 1,500 rows — final evaluation (used exactly once per model)
        """)

    if features:
        st.divider()
        st.markdown(f"### All {len(features)} Features")
        # display in 3 columns
        n = len(features)
        cols = st.columns(3)
        chunk = (n + 2) // 3
        for i, col in enumerate(cols):
            with col:
                chunk_features = features[i*chunk:(i+1)*chunk]
                for f in chunk_features:
                    st.markdown(f"- `{f}`")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison")

    if all_results.empty:
        st.warning("No results found. Run all member notebooks first to generate the comparison CSVs.")
    else:
        # KPI cards — best primary model
        dt_mask = all_results["Model"].str.contains("Decision Tree", case=False)
        primary_df  = all_results[~dt_mask].copy()
        baseline_df = all_results[dt_mask].copy()

        best_idx = primary_df["F1_Score"].idxmax()
        best_row = primary_df.loc[best_idx]

        # Shorten long model names for metric cards
        def short_name(name):
            name_map = {
                "Random Forest": "Rnd Forest", "XGBoost": "XGBoost",
                "Support Vector Machine": "SVM", "K-Nearest Neighbours": "KNN",
                "LR+Poly": "LR+Poly", "Multi-Layer Perceptron": "MLP",
            }
            for k, v in name_map.items():
                if k.lower() in str(name).lower():
                    return v
            return str(name)[:14]

        st.markdown("### Best Primary Model (Test F1)")
        c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
        c1.metric("Model", short_name(best_row["Model"]))
        c2.metric("F1-Score", f"{best_row['F1_Score']:.4f}")
        c3.metric("ROC-AUC", f"{best_row['ROC_AUC']:.4f}")
        c4.metric("MCC", f"{best_row['MCC']:.4f}")
        c5.metric("Overfit Gap", f"{best_row['Overfit_Gap']:.4f}", delta=f"{best_row['Overfit_Label']}")

        st.divider()

        # Metric selector
        metric = st.selectbox(
            "Select metric to visualise",
            ["F1_Score", "ROC_AUC", "MCC", "Overfit_Gap", "FP_Cost_GBP"],
            format_func=lambda x: {
                "F1_Score": "F1-Score", "ROC_AUC": "ROC-AUC",
                "MCC": "MCC", "Overfit_Gap": "Overfit Gap",
                "FP_Cost_GBP": "FP Cost (£)"
            }.get(x, x)
        )

        fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
        x = np.arange(len(all_results))
        colors = [bar_color(m) for m in all_results["Model"]]
        scale = 1/1000 if metric == "FP_Cost_GBP" else 1
        vals = all_results[metric] * scale if metric in all_results.columns else None

        if vals is not None:
            bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + vals.max() * 0.01,
                        f"{val:.3f}" if metric != "FP_Cost_GBP" else f"£{val:.0f}k",
                        ha="center", va="bottom", fontsize=8)

        _member_short = {
            "RF_DT_Gokulkrishna": "M1",
            "XGBoost_DT_Joel":    "M2",
            "SVM_DT_Apprajit":    "M3",
            "LRPoly_DT_Afzal":    "M4",
            "MLP_DT_Vino":        "M5",
        }
        def axis_label(row):
            m = _member_short.get(row["Member"], row["Member"][:4])
            if "Decision Tree" in str(row["Model"]):
                return f"{m}\nDT"
            first = row["Model"].split(" ")[0][:6]
            return f"{m}\n{first}"

        labels = [axis_label(row) for _, row in all_results.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, ha="center")
        ylabel = {"F1_Score": "F1-Score", "ROC_AUC": "ROC-AUC",
                  "MCC": "MCC", "Overfit_Gap": "Overfit Gap",
                  "FP_Cost_GBP": "FP Cost (£k)"}.get(metric, metric)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — All Models", fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

        # legend
        legend_patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
        ax.legend(handles=legend_patches, fontsize=7, loc="upper right", ncol=2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.markdown("### Full Comparison Table")
        display_cols = ["Member", "Model", "F1_Score", "ROC_AUC", "MCC",
                        "TNR", "FPR", "Overfit_Gap", "Overfit_Label"]
        available = [c for c in display_cols if c in all_results.columns]
        st.dataframe(all_results[available].reset_index(drop=True), use_container_width=True)

        st.divider()
        st.markdown("### Primary Model vs Member Decision Tree: F1 Delta")
        if len(primary_df) == len(baseline_df):
            delta_df = primary_df[["Member", "Model", "F1_Score"]].copy()
            delta_df["DT_F1"]    = baseline_df["F1_Score"].values
            delta_df["F1_Delta"] = (delta_df["F1_Score"] - delta_df["DT_F1"]).round(4)

            fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=100)
            x2 = np.arange(len(delta_df))
            bar_colors2 = ["#2DC653" if v > 0 else "#E84855" for v in delta_df["F1_Delta"]]
            ax2.bar(x2, delta_df["F1_Delta"], color=bar_colors2, edgecolor="white")
            ax2.axhline(0, color="black", lw=0.8)
            for i, val in enumerate(delta_df["F1_Delta"]):
                ax2.text(i, val + 0.001, f"{val:+.4f}", ha="center", va="bottom", fontsize=9)
            _ms = {"RF_DT_Gokulkrishna":"M1","XGBoost_DT_Joel":"M2",
                    "SVM_DT_Apprajit":"M3","LRPoly_DT_Afzal":"M4","MLP_DT_Vino":"M5"}
            labels2 = [f"{_ms.get(row['Member'], row['Member'][:4])}\n{row['Model'].split(' ')[0]}"
                       for _, row in delta_df.iterrows()]
            ax2.set_xticks(x2)
            ax2.set_xticklabels(labels2, fontsize=9)
            ax2.set_ylabel("F1 Delta (Primary – Member DT)")
            ax2.set_title("F1 Improvement over Each Member's Decision Tree", fontweight="bold")
            ax2.grid(True, alpha=0.2, axis="y")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Group-level figures generated by Group_Combined_Comparison.ipynb ─
        st.divider()
        st.markdown("### Cross-Model Permutation Importance")
        _perm_path = BASE / "group_perm_importance_heatmap.png"
        if _perm_path.exists():
            st.image(str(_perm_path), caption=(
                "Permutation importance across all five primary models (test set). "
                "Darker cells indicate higher importance. "
                "Load_Factor and price_per_km rank in the top 3 for every model."
            ), use_container_width=True)
        else:
            st.info(
                "group_perm_importance_heatmap.png not yet generated. "
                "Run Section 12 of Group_Combined_Comparison.ipynb to produce it."
            )

        st.divider()
        st.markdown("### F1-Score vs Decision Threshold (All Primary Models)")
        _thresh_path = BASE / "group_threshold_comparison.png"
        if _thresh_path.exists():
            st.image(str(_thresh_path), caption=(
                "F1-score vs decision threshold for all five primary models "
                "on the held-out test set. "
                "The SVM's peak shifts right, reflecting its high-precision, lower-recall profile."
            ), use_container_width=True)
        else:
            st.info(
                "group_threshold_comparison.png not yet generated. "
                "Run Section 13 of Group_Combined_Comparison.ipynb to produce it."
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Member Deep-Dive
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Member Deep-Dive":
    st.title("Member Deep-Dive")

    member = st.selectbox("Select member", list(M_DIRS.keys()))
    m_dir = M_DIRS[member]

    if all_results.empty or member not in all_results["Member"].values:
        st.warning(f"No results for {member}. Run the member notebook first.")
    else:
        member_data = all_results[all_results["Member"] == member].copy()
        dt_mask     = member_data["Model"].str.contains("Decision Tree", case=False)
        primary_row = member_data[~dt_mask].iloc[0] if (~dt_mask).any() else None
        dt_row      = member_data[dt_mask].iloc[0] if dt_mask.any() else None

        if primary_row is not None:
            _member_display = {
                "RF_DT_Gokulkrishna": "Gokulkrishna (RF)",
                "XGBoost_DT_Joel":    "Joel (XGBoost)",
                "SVM_DT_Apprajit":    "Apprajit (SVM)",
                "LRPoly_DT_Afzal":    "Afzal (LR+Poly)",
                "MLP_DT_Vino":        "Vino (MLP)",
            }
            st.subheader(f"{_member_display.get(member, member)} — {primary_row['Model']}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("F1-Score (Test)", f"{primary_row['F1_Score']:.4f}")
            c2.metric("ROC-AUC", f"{primary_row['ROC_AUC']:.4f}")
            c3.metric("MCC", f"{primary_row['MCC']:.4f}")
            c4.metric("Overfit Gap", f"{primary_row['Overfit_Gap']:.4f}",
                      delta=primary_row.get("Overfit_Label", ""))

        if primary_row is not None and dt_row is not None:
            st.divider()
            st.markdown("### Primary Model vs Decision Tree")
            comp = pd.DataFrame([primary_row, dt_row])
            display_cols = ["Model", "F1_Score", "ROC_AUC", "MCC", "TNR",
                            "FPR", "Overfit_Gap", "Overfit_Label"]
            available = [c for c in display_cols if c in comp.columns]
            st.dataframe(comp[available].reset_index(drop=True), use_container_width=True)

            # Delta bar
            delta_f1  = float(primary_row["F1_Score"]) - float(dt_row["F1_Score"])
            delta_auc = float(primary_row["ROC_AUC"])  - float(dt_row["ROC_AUC"])

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
            for ax, (metric, delta, label) in zip(
                axes,
                [("F1_Score", delta_f1, "F1-Score"), ("ROC_AUC", delta_auc, "ROC-AUC")]
            ):
                names  = [str(primary_row["Model"]), "Decision Tree"]
                values = [float(primary_row[metric]), float(dt_row[metric])]
                colors = [bar_color(names[0]), "#7F8C8D"]
                bars = ax.bar(names, values, color=colors, edgecolor="white")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel(label)
                ax.set_title(f"{label} — {_member_display.get(member, member)}", fontweight="bold")
                ax.set_ylim(min(values) * 0.98, 1.0)
                ax.grid(True, alpha=0.2, axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Show saved figures if present
        st.divider()
        st.markdown("### Saved Visualisations")

        # Map member to figure prefix
        suffix_map = {
            "RF_DT_Gokulkrishna": "gokulkrishna", "XGBoost_DT_Joel": "joel",
            "SVM_DT_Apprajit":    "apprajit",      "LRPoly_DT_Afzal": "afzal",
            "MLP_DT_Vino":        "vino",
        }
        suffix = suffix_map.get(member, "")
        model_prefix_map = {
            "RF_DT_Gokulkrishna": "rf",  "XGBoost_DT_Joel": "xgb",
            "SVM_DT_Apprajit":    "svm", "LRPoly_DT_Afzal": "lr",
            "MLP_DT_Vino":        "mlp",
        }
        mp = model_prefix_map.get(member, "")

        # Figures are saved inside figures/ subdirectory of each member folder
        fig_dir = m_dir / "figures"
        possible_figs = [
            f"{mp}_tuned_learning_curve_{suffix}.png",
            f"{mp}_feature_importance_{suffix}.png",
            f"perm_importance_{suffix}.png",
            f"{mp}_perm_importance_{suffix}.png",   # RF stores as rf_perm_importance_...
            f"{mp}_loss_curve_{suffix}.png",
            f"{mp}_logloss_curve_{suffix}.png",     # XGBoost log-loss variant
            f"{mp}_threshold_sweep_{suffix}.png",
            f"{mp}_cm_roc_test_{suffix}.png",
            f"dt_cm_roc_test_{suffix}.png",
            f"dt_tree_viz_{suffix}.png",
            f"dt_threshold_sweep_{suffix}.png",
            f"dt_tuned_learning_curve_{suffix}.png",
        ]

        # Search in figures/ subdirectory first, then fall back to member root
        search_dir = fig_dir if fig_dir.exists() else m_dir
        found = [(f, search_dir / f) for f in possible_figs if (search_dir / f).exists()]
        if found:
            for name, path in found:
                st.image(str(path), caption=name)
        else:
            st.info("No saved figures found. Run the member notebook to generate them.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Business Insights
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Business Insights":
    st.title("Business Insights")
    st.subheader("From Prediction to Action")

    if not all_results.empty:
        dt_mask = all_results["Model"].str.contains("Decision Tree", case=False)
        primary = all_results[~dt_mask].sort_values("F1_Score", ascending=False)
        top = primary.iloc[0] if len(primary) > 0 else None

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Primary Model F1", f"{top['F1_Score']:.4f}" if top is not None else "—")
            st.metric("Average Primary F1", f"{primary['F1_Score'].mean():.4f}")
        with col2:
            if top is not None and not pd.isna(top.get("FP_Cost_GBP", float("nan"))):
                st.metric("Best Model FP Cost", f"£{int(top['FP_Cost_GBP']):,}")
            st.metric("Models Evaluated", str(len(primary)))

    st.divider()

    st.markdown("""
    ### Key Operational Findings

    **1. Load Factor is the primary driver of profitability**
    Routes operating below ~65% passenger capacity are systematically loss-making across all models.
    *Action: Flag routes with consistent Load_Factor < 0.65 for restructuring or retirement.*

    **2. Price per km matters more than absolute price**
    High ticket prices on short routes do not guarantee profitability if operating costs are proportionally higher.
    *Action: Use price_per_km as a KPI alongside absolute revenue when evaluating route performance.*

    **3. Season amplifies all other effects**
    Peak season shifts profitability probability significantly even for marginal routes.
    *Action: Consider seasonal route licences for borderline routes — operate only in Peak season.*

    **4. Competition Index affects profitability on high-competition routes**
    Routes with high competition force lower prices, reducing margins.
    *Action: Prioritise less competitive routes for new launches; monitor market entries on existing routes.*

    ---

    ### Decision Threshold Guidance

    All models provide probability outputs, not just binary labels. The default threshold of **0.5** is not
    necessarily optimal for business decisions.

    | Airline Priority | Recommended Threshold | Effect |
    |-----------------|----------------------|--------|
    | Minimise FP (avoid bad routes) | Raise threshold (>0.6) | Fewer unprofitable routes approved; some profitable routes missed |
    | Maximise recall (catch all profitable routes) | Lower threshold (<0.4) | More profitable routes found; more unprofitable routes accepted |
    | Minimise total £ cost | Use cost-optimal threshold from sweep | Balances £42k FP vs £30k FN |

    ---

    ### Model Deployment Recommendations

    | Use Case | Best Model | Reason |
    |----------|-----------|--------|
    | High-accuracy route scoring | Ensemble (RF/XGBoost) | Highest F1, manageable overfit gap |
    | Board-level explainability | Decision Tree | Direct rule visualisation |
    | Real-time API endpoint | MLP or LR+Poly | Fast inference |
    | Low-data new markets | SVM (RBF) | Max-margin generalisation from few examples |

    ---

    ### Study Limitations

    - **Synthetic data**: All results are based on 15,000 synthetic records. Real airline data contains
      seasonal drift, irregular events (pandemics, strikes), and operational noise not present here.
    - **Cost estimates**: £42k FP and £30k FN are order-of-magnitude figures. A sensitivity analysis
      across a range of cost assumptions is recommended before production deployment.
    - **Static features**: The model uses point-in-time features. A production system should incorporate
      time-series features (trailing load factor trend, price trajectory) for better predictive power.
    """)

    if not all_results.empty:
        st.divider()
        st.markdown("### Final Model Rankings")

        rank_cols = ["Member", "Model", "F1_Score", "ROC_AUC", "MCC", "Overfit_Gap", "Overfit_Label"]
        available_cols = [c for c in rank_cols if c in all_results.columns]

        dt_mask = all_results["Model"].str.contains("Decision Tree", case=False)

        st.markdown("**Primary Models** (ranked by F1-Score)")
        primary_ranked = all_results[~dt_mask][available_cols].copy()
        primary_ranked = primary_ranked.sort_values("F1_Score", ascending=False).reset_index(drop=True)
        primary_ranked.index += 1
        st.dataframe(primary_ranked, use_container_width=True)

        st.markdown("**Decision Trees** — independently tuned per member")
        dt_ranked = all_results[dt_mask][available_cols].copy()
        dt_ranked = dt_ranked.sort_values("F1_Score", ascending=False).reset_index(drop=True)
        dt_ranked.index += 1
        st.dataframe(dt_ranked, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — ML Prediction
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ML Prediction":
    import pickle
    import warnings
    warnings.filterwarnings("ignore")

    st.title("ML Prediction — Route Profitability")
    st.markdown(
        "Adjust the route parameters below and get an instant profitability prediction "
        "from any of the ten trained models (5 primary + 5 independently tuned Decision Trees). "
        "Start with a preset scenario, then fine-tune."
    )

    # ── Model PKL map ─────────────────────────────────────────────────────
    MODEL_PKLS = {
        # Primary models
        "Random Forest"     : BASE / "RF_DT_Gokulkrishna" / "best_rf_gokulkrishna.pkl",
        "XGBoost"           : BASE / "XGBoost_DT_Joel"    / "best_xgb_joel.pkl",
        "SVM RBF"           : BASE / "SVM_DT_Apprajit"    / "best_svm_apprajit.pkl",
        "LR+Poly"           : BASE / "LRPoly_DT_Afzal"    / "best_lrpoly_afzal.pkl",
        "MLP"               : BASE / "MLP_DT_Vino"        / "best_mlp_vino.pkl",
        # Decision Trees — independently tuned per member
        "DT (Gokulkrishna)" : BASE / "RF_DT_Gokulkrishna" / "best_dt_gokulkrishna.pkl",
        "DT (Joel)"         : BASE / "XGBoost_DT_Joel"    / "best_dt_joel.pkl",
        "DT (Apprajit)"     : BASE / "SVM_DT_Apprajit"    / "best_dt_apprajit.pkl",
        "DT (Afzal)"        : BASE / "LRPoly_DT_Afzal"    / "best_dt_afzal.pkl",
        "DT (Vino)"         : BASE / "MLP_DT_Vino"        / "best_dt_vino.pkl",
    }
    TREE_MODELS   = {                                  # trained on unscaled data
        "Random Forest", "XGBoost",
        "DT (Gokulkrishna)", "DT (Joel)", "DT (Apprajit)", "DT (Afzal)", "DT (Vino)",
    }
    SCALED_MODELS = {"SVM RBF", "LR+Poly", "MLP"}     # trained on scaled data

    @st.cache_resource
    def _load_scaler():
        p = PROC / "scaler.pkl"
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    @st.cache_resource
    def _load_model(name):
        p = MODEL_PKLS[name]
        if not p.exists():
            return None, f"PKL not found: {p.name}"
        try:
            with open(p, "rb") as f:
                return pickle.load(f), None
        except Exception as e:
            return None, str(e)

    _scaler    = _load_scaler()
    _feat_list = load_feature_names()

    # ── Presets (all values in raw user-friendly space) ───────────────────
    _PRESETS = {
        # HIGH: load=0.90, ppk = log1p(1500)/(log1p(5000)+1) = 7.313/9.517 = 0.768 (high end)
        "high": dict(
            load_factor=0.90, avg_ticket=1500.0, flight_dist=5000.0,
            on_time=92.0, satisfaction=9.2, competition=2,
            season="Peak", demand="High", route_cat="Medium Haul",
            alliance="Star Alliance", origin="Europe", dest="North America",
            aircraft_cap=320, aircraft_age=7, passengers=288,
            flight_hours=8.0, weather=0, delay_min=0.0,
            fuel_price=0.65, market_share=22.0, is_narrow=0, is_weekend=0, month=7,
        ),
        # AVERAGE: medians — ppk = log1p(656)/(log1p(5993)+1) = 6.487/9.696 = 0.669
        "average": dict(
            load_factor=0.79, avg_ticket=656.0, flight_dist=5993.0,
            on_time=82.0, satisfaction=8.3, competition=5,
            season="Shoulder", demand="Medium", route_cat="Long Haul",
            alliance="No Alliance", origin="Africa / Asia (default)", dest="Africa (default)",
            aircraft_cap=296, aircraft_age=11, passengers=230,
            flight_hours=7.8, weather=0, delay_min=0.0,
            fuel_price=0.72, market_share=16.8, is_narrow=0, is_weekend=0, month=7,
        ),
        # STRUGGLING: load=0.52, ppk = log1p(80)/(log1p(10000)+1) = 4.394/10.211 = 0.430 (low end)
        "struggling": dict(
            load_factor=0.52, avg_ticket=80.0, flight_dist=10000.0,
            on_time=64.0, satisfaction=7.2, competition=8,
            season="Low", demand="Low", route_cat="Long Haul",
            alliance="No Alliance", origin="Africa / Asia (default)", dest="Africa (default)",
            aircraft_cap=250, aircraft_age=16, passengers=130,
            flight_hours=14.0, weather=1, delay_min=45.0,
            fuel_price=0.88, market_share=5.0, is_narrow=0, is_weekend=0, month=1,
        ),
    }

    if "pred_preset" not in st.session_state:
        st.session_state.pred_preset = _PRESETS["average"].copy()

    # ── Model selector ────────────────────────────────────────────────────
    _col_sel, _col_badge = st.columns([1, 2])
    with _col_sel:
        _sel_model = st.selectbox("Select Model", list(MODEL_PKLS.keys()))
    with _col_badge:
        _mc = MODEL_COLORS.get(_sel_model, "#95A5A6")
        _DT_MEMBER_NOTE = {
            "DT (Gokulkrishna)": "Decision Tree — Gokulkrishna's tuned parameters",
            "DT (Joel)":         "Decision Tree — Joel's tuned parameters",
            "DT (Apprajit)":     "Decision Tree — Apprajit's tuned parameters",
            "DT (Afzal)":        "Decision Tree — Afzal's tuned parameters",
            "DT (Vino)":         "Decision Tree — Vino's tuned parameters",
        }
        if _sel_model in _DT_MEMBER_NOTE:
            _scale_note = _DT_MEMBER_NOTE[_sel_model] + " · unscaled features (tree-based)"
        elif _sel_model in TREE_MODELS:
            _scale_note = "unscaled features (tree-based)"
        else:
            _scale_note = "scaled features (distance/linear)"
        st.markdown(
            f'<div style="background:{_mc}22;border-left:4px solid {_mc};'
            f'padding:10px;border-radius:4px;margin-top:28px">'
            f'<b>{_sel_model}</b> — {_scale_note}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Scenario preset buttons ───────────────────────────────────────────
    st.markdown("#### Step 1 — Pick a Starting Scenario")
    _pc1, _pc2, _pc3 = st.columns(3)
    with _pc1:
        if st.button("🟢 High-Performing Route", use_container_width=True):
            st.session_state.pred_preset = _PRESETS["high"].copy()
            st.rerun()
    with _pc2:
        if st.button("🟡 Average Route", use_container_width=True):
            st.session_state.pred_preset = _PRESETS["average"].copy()
            st.rerun()
    with _pc3:
        if st.button("🔴 Struggling Route", use_container_width=True):
            st.session_state.pred_preset = _PRESETS["struggling"].copy()
            st.rerun()

    _pv = st.session_state.pred_preset

    st.divider()
    st.markdown("#### Step 2 — Fine-Tune Route Parameters")

    # ── Main widgets — left and right columns ─────────────────────────────
    _wl, _wr = st.columns(2)

    with _wl:
        st.markdown("**Key Operational Metrics**")
        _load_factor  = st.slider(
            "Load Factor (passenger fill rate)",
            0.40, 1.00, float(_pv["load_factor"]), step=0.01,
            help="Proportion of aircraft seats occupied. Primary profitability driver.",
        )
        _avg_ticket = st.number_input(
            "Average Ticket Price (£)", min_value=30.0, max_value=3500.0,
            value=float(_pv["avg_ticket"]), step=10.0,
            help="Mean revenue per passenger ticket. Higher = better yield.",
        )
        _flight_dist = st.number_input(
            "Flight Distance (km)", min_value=300.0, max_value=16000.0,
            value=float(_pv["flight_dist"]), step=50.0,
            help="Great-circle distance of the route.",
        )
        _on_time = st.slider(
            "On-Time Performance (%)", 55.0, 99.0, float(_pv["on_time"]), step=0.5,
        )
        _satisfaction = st.slider(
            "Passenger Satisfaction Score", 6.7, 10.0, float(_pv["satisfaction"]), step=0.1,
        )
        _competition = st.slider(
            "Competition Index (1 = low, 9 = high)", 1, 9, int(_pv["competition"]),
        )

    with _wr:
        st.markdown("**Route & Market Context**")

        _season_map  = {"Low": 0, "Normal": 1, "Shoulder": 2, "Peak": 3}
        _season_keys = list(_season_map.keys())
        _season      = st.selectbox(
            "Season",
            _season_keys,
            index=_season_keys.index(_pv["season"]) if _pv["season"] in _season_keys else 2,
        )

        _demand_map  = {"Low": 0, "Medium": 1, "High": 2}
        _demand_keys = list(_demand_map.keys())
        _demand      = st.selectbox(
            "Demand Level",
            _demand_keys,
            index=_demand_keys.index(_pv["demand"]) if _pv["demand"] in _demand_keys else 1,
        )

        _route_opts = ["Long Haul", "Medium Haul", "Short Haul"]
        _route_cat  = st.selectbox(
            "Route Category",
            _route_opts,
            index=_route_opts.index(_pv["route_cat"]) if _pv["route_cat"] in _route_opts else 0,
            help="Long Haul >6h · Medium Haul 3-6h · Short Haul <3h",
        )

        _alliance_opts = ["No Alliance", "OneWorld", "SkyTeam", "Star Alliance"]
        _alliance      = st.selectbox(
            "Airline Alliance",
            _alliance_opts,
            index=_alliance_opts.index(_pv["alliance"]) if _pv["alliance"] in _alliance_opts else 0,
        )

        _origin_opts = [
            "Africa / Asia (default)", "Europe", "Middle East",
            "North America", "Oceania",
        ]
        _origin = st.selectbox(
            "From (Origin Region)",
            _origin_opts,
            index=_origin_opts.index(_pv["origin"]) if _pv["origin"] in _origin_opts else 0,
            help="Origin region drives one-hot encoded origin_region features.",
        )

        _dest_opts = [
            "Africa (default)", "Asia", "Europe", "Middle East",
            "North America", "Oceania", "Other", "South America",
        ]
        _dest = st.selectbox(
            "To (Destination Region)",
            _dest_opts,
            index=_dest_opts.index(_pv["dest"]) if _pv["dest"] in _dest_opts else 0,
        )

    # ── Advanced expander ─────────────────────────────────────────────────
    with st.expander("Advanced / Optional Features (defaults are dataset medians)"):
        _av1, _av2 = st.columns(2)
        with _av1:
            _aircraft_cap  = st.number_input("Aircraft Capacity (seats)", 180, 517,
                                              int(_pv["aircraft_cap"]), step=1)
            _aircraft_age  = st.number_input("Aircraft Age (years)", 1, 21,
                                              int(_pv["aircraft_age"]), step=1)
            _passengers    = st.number_input("Passengers on Flight", 76, 422,
                                              int(_pv["passengers"]), step=1)
            _flight_hours  = st.number_input("Flight Hours", 0.8, 21.0,
                                              float(_pv["flight_hours"]), step=0.1)
            _delay_min     = st.number_input("Delay Minutes", 0.0, 117.0,
                                              float(_pv["delay_min"]), step=1.0)
        with _av2:
            _fuel_price    = st.number_input("Fuel Price per Litre (£)", 0.44, 1.04,
                                              float(_pv["fuel_price"]), step=0.01)
            _market_share  = st.number_input("Market Share (%)", 3.0, 79.9,
                                              float(_pv["market_share"]), step=0.5)
            _weather       = st.selectbox(
                "Weather Disruption",
                [0, 1], index=int(_pv["weather"]),
                format_func=lambda x: "No disruption" if x == 0 else "Disruption present",
            )
            _is_narrow     = st.selectbox(
                "Aircraft Body Type",
                [0, 1], index=int(_pv["is_narrow"]),
                format_func=lambda x: "Wide-body" if x == 0 else "Narrow-body (737 / A320 / A321)",
            )
            _is_weekend    = st.selectbox(
                "Weekend Flight",
                [0, 1], index=int(_pv["is_weekend"]),
                format_func=lambda x: "Weekday" if x == 0 else "Weekend",
            )
            _month         = st.slider("Flight Month", 1, 12, int(_pv["month"]))

    st.divider()

    # ── Predict ───────────────────────────────────────────────────────────
    if st.button("🔮  Predict Route Profitability", type="primary", use_container_width=True):

        import numpy as _np

        # log1p transforms (these 3 columns were log1p-transformed during preprocessing)
        _fd_log    = _np.log1p(_flight_dist)
        _atp_log   = _np.log1p(_avg_ticket)
        _delay_log = _np.log1p(_delay_min)

        # price_per_km was computed AFTER log1p in the preprocessing notebook:
        #   df['price_per_km'] = df['Average_Ticket_Price'] / (df['Flight_Distance_KM'] + 1)
        # where both columns were already in log1p space at that point.
        _price_per_km   = _atp_log / (_fd_log + 1) if _fd_log > 0 else 0.0
        _load_rev_ratio = _load_factor * _price_per_km
        _delay_flag     = 1.0 if _delay_min > 30 else 0.0

        # Cyclical season encoding
        _s_ord      = float(_season_map[_season])
        _d_ord      = float(_demand_map[_demand])
        _season_sin = _np.sin(2 * _np.pi * _s_ord / 4)
        _season_cos = _np.cos(2 * _np.pi * _s_ord / 4)

        # OHE: Route Category (reference = Long Haul)
        _rc_med   = 1.0 if _route_cat == "Medium Haul" else 0.0
        _rc_short = 1.0 if _route_cat == "Short Haul"  else 0.0

        # OHE: Alliance (reference = No Alliance)
        _al_ow = 1.0 if _alliance == "OneWorld"      else 0.0
        _al_st = 1.0 if _alliance == "SkyTeam"       else 0.0
        _al_sa = 1.0 if _alliance == "Star Alliance" else 0.0

        # OHE: Origin region (reference = Africa / Asia)
        _or_eu = 1.0 if _origin == "Europe"        else 0.0
        _or_me = 1.0 if _origin == "Middle East"   else 0.0
        _or_na = 1.0 if _origin == "North America" else 0.0
        _or_oc = 1.0 if _origin == "Oceania"       else 0.0

        # OHE: Destination region (reference = Africa)
        _de_as = 1.0 if _dest == "Asia"          else 0.0
        _de_eu = 1.0 if _dest == "Europe"        else 0.0
        _de_me = 1.0 if _dest == "Middle East"   else 0.0
        _de_na = 1.0 if _dest == "North America" else 0.0
        _de_oc = 1.0 if _dest == "Oceania"       else 0.0
        _de_ot = 1.0 if _dest == "Other"         else 0.0
        _de_sa = 1.0 if _dest == "South America" else 0.0

        # Assemble feature dict in exact order from feature_names.json
        _fdict = {
            "Flight_Distance_KM":        _fd_log,
            "Aircraft_Capacity":         float(_aircraft_cap),
            "Aircraft_Age_Years":        float(_aircraft_age),
            "Passengers":                float(_passengers),
            "Load_Factor":               float(_load_factor),
            "Flight_Hours":              float(_flight_hours),
            "Competition_Index":         float(_competition),
            "Weather_Disruption":        float(_weather),
            "On_Time_Performance":       float(_on_time),
            "Delay_Minutes":             _delay_log,
            "Average_Ticket_Price":      _atp_log,
            "Fuel_Price_Per_Liter":      float(_fuel_price),
            "Passenger_Satisfaction_Score": float(_satisfaction),
            "Market_Share_Pct":          float(_market_share),
            "price_per_km":              _price_per_km,
            "delay_flag":                _delay_flag,
            "flight_month":              float(_month),
            "is_weekend":                float(_is_weekend),
            "is_narrow_body":            float(_is_narrow),
            "load_revenue_ratio":        _load_rev_ratio,
            "Season_Ordinal":            _s_ord,
            "Demand_Ordinal":            _d_ord,
            "season_sin":                _season_sin,
            "season_cos":                _season_cos,
            "Route_Category_Medium Haul": _rc_med,
            "Route_Category_Short Haul":  _rc_short,
            "Alliance_OneWorld":          _al_ow,
            "Alliance_SkyTeam":           _al_st,
            "Alliance_Star Alliance":     _al_sa,
            "origin_region_Europe":       _or_eu,
            "origin_region_Middle_East":  _or_me,
            "origin_region_North_America": _or_na,
            "origin_region_Oceania":      _or_oc,
            "dest_region_Asia":           _de_as,
            "dest_region_Europe":         _de_eu,
            "dest_region_Middle_East":    _de_me,
            "dest_region_North_America":  _de_na,
            "dest_region_Oceania":        _de_oc,
            "dest_region_Other":          _de_ot,
            "dest_region_South_America":  _de_sa,
        }

        if not _feat_list:
            st.error("feature_names.json not found in processed_data/. Run the preprocessing notebook first.")
        else:
            _X_in = _np.array([[_fdict[f] for f in _feat_list]])

            # Apply scaler for distance/linear models
            if _sel_model in SCALED_MODELS and _scaler is not None:
                _X_model = _scaler.transform(_X_in)
            else:
                _X_model = _X_in

            _model, _err = _load_model(_sel_model)
            if _err:
                st.error(f"Could not load model: {_err}")
            else:
                try:
                    _proba = _model.predict_proba(_X_model)[0, 1]
                    _pred  = int(_proba >= 0.5)

                    st.divider()
                    st.markdown("### Prediction Result")

                    # Result banner
                    if _pred == 1:
                        st.success(f"### ✅  PROFITABLE   —   Confidence: {_proba:.1%}")
                    else:
                        st.error(f"### ❌  LOSS-MAKING   —   Profit probability: {_proba:.1%}")

                    # Probability gauge bar
                    _fig_g, _ax_g = plt.subplots(figsize=(9, 1.0), dpi=100)
                    _bar_col = "#2DC653" if _pred == 1 else "#E84855"
                    _ax_g.barh([""], [_proba],       color=_bar_col,   height=0.55)
                    _ax_g.barh([""], [1 - _proba],   color="#e8e8e8",  height=0.55, left=[_proba])
                    _ax_g.axvline(0.5, color="black", lw=1.5, ls="--", alpha=0.7)
                    _ax_g.set_xlim(0, 1)
                    _ax_g.set_xlabel("Probability of Profitability →", fontsize=9)
                    _ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                    _ax_g.set_xticklabels(["0%", "25%", "50% (threshold)", "75%", "100%"], fontsize=8)
                    _ax_g.set_yticks([])
                    _lbl_x = _proba / 2 if _proba > 0.12 else _proba + 0.06
                    _ax_g.text(_lbl_x, 0, f"{_proba:.1%}", ha="center", va="center",
                               fontweight="bold", color="white", fontsize=10)
                    _fig_g.tight_layout()
                    st.pyplot(_fig_g)
                    plt.close(_fig_g)

                    # Key inputs summary cards
                    st.markdown("**Inputs summary:**")
                    _k1, _k2, _k3, _k4, _k5, _k6 = st.columns(6)
                    _k1.metric("Load Factor",    f"{_load_factor:.2f}")
                    _k2.metric("Ticket Price",   f"£{_avg_ticket:.0f}")
                    _k3.metric("Distance",       f"{_flight_dist:.0f} km")
                    _k4.metric("On-Time",        f"{_on_time:.0f}%")
                    _k5.metric("Season",         _season)
                    _k6.metric("price/km",       f"£{_price_per_km:.3f}")

                    # Auto-derived features info
                    with st.expander("Show computed intermediate features"):
                        _ic1, _ic2 = st.columns(2)
                        with _ic1:
                            st.write(f"**Flight_Distance_KM** (log1p) = log1p({_flight_dist:.0f}) = **{_fd_log:.4f}**")
                            st.write(f"**Average_Ticket_Price** (log1p) = log1p({_avg_ticket:.0f}) = **{_atp_log:.4f}**")
                            st.write(f"**Delay_Minutes** (log1p) = log1p({_delay_min:.1f}) = **{_delay_log:.4f}**")
                            st.write(f"**price_per_km** = {_atp_log:.4f} ÷ ({_fd_log:.4f}+1) = **{_price_per_km:.4f}**")
                        with _ic2:
                            st.write(f"**load_revenue_ratio** = {_load_factor:.2f} × {_price_per_km:.4f} = **{_load_rev_ratio:.4f}**")
                            st.write(f"**delay_flag** = {'1 (delayed >30 min)' if _delay_flag else '0 (on time)'}")
                            st.write(f"**season_sin / cos** = {_season_sin:.4f} / {_season_cos:.4f}")
                            st.write(f"**Season_Ordinal** = {_s_ord:.0f} · **Demand_Ordinal** = {_d_ord:.0f}")

                except Exception as _pred_err:
                    st.error(f"Prediction failed: {_pred_err}")
