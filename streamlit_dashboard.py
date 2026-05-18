"""
Airline Route Profitability — Interactive Dashboard
COM3032/COMM074 · Group Project

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
import joblib
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
    "Member 1": BASE / "Member 1",
    "Member 2": BASE / "Member 2",
    "Member 3": BASE / "Member 3",
    "Member 4": BASE / "Member 4",
    "Member 5": BASE / "Member 5",
}
CSV_MAP = {
    "Member 1": M_DIRS["Member 1"] / "member1_model_comparison.csv",
    "Member 2": M_DIRS["Member 2"] / "member2_model_comparison.csv",
    "Member 3": M_DIRS["Member 3"] / "member3_model_comparison.csv",
    "Member 4": M_DIRS["Member 4"] / "member4_model_comparison.csv",
    "Member 5": M_DIRS["Member 5"] / "member5_model_comparison.csv",
}

MODEL_COLORS = {
    "Random Forest" : "#2E86AB",
    "XGBoost"       : "#F6AE2D",
    "SVM RBF"       : "#9B59B6",
    "KNN"           : "#E67E22",
    "MLP"           : "#E74C3C",
    "Decision Tree" : "#7F8C8D",
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
            "Profitability Predictor",
        ],
    )
    st.divider()
    st.caption("COM3032/COMM074 · Airline Route Profitability")
    st.caption("Group Project · 2025–26")

all_results = load_results()
stats       = load_preprocessing_stats()
features    = load_feature_names()


# ── Real trained-model prediction helpers ─────────────────────────────────
MODEL_PATHS = {
    "Random Forest": M_DIRS["Member 1"] / "best_rf_model_m1.pkl",
    "XGBoost": M_DIRS["Member 2"] / "best_xgb_model_m2.pkl",
    "SVM RBF": M_DIRS["Member 3"] / "best_svm_model_m3.pkl",
    "KNN": M_DIRS["Member 4"] / "best_knn_model_m4.pkl",
    "MLP": M_DIRS["Member 5"] / "best_mlp_model_m5.pkl",
    "Decision Tree - Member 1": M_DIRS["Member 1"] / "best_dt_model_m1.pkl",
    "Decision Tree - Member 2": M_DIRS["Member 2"] / "best_dt_model_m2.pkl",
    "Decision Tree - Member 3": M_DIRS["Member 3"] / "best_dt_model_m3.pkl",
    "Decision Tree - Member 4": M_DIRS["Member 4"] / "best_dt_model_m4.pkl",
    "Decision Tree - Member 5": M_DIRS["Member 5"] / "best_dt_model_m5.pkl",
}

@st.cache_resource
def load_trained_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if path is None or not path.exists():
        return None, path
    return joblib.load(path), path

@st.cache_resource
def load_scaler():
    path = PROC / "scaler.pkl"
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_data
def load_training_defaults():
    """Use training medians as safe defaults for features not entered manually."""
    path = PROC / "X_train.csv"
    if path.exists():
        train_df = pd.read_csv(path)
        return train_df.median(numeric_only=True).to_dict()
    return {}


def _set_if_exists(row, names, value):
    for name in names:
        if name in row.index:
            row[name] = value


def build_prediction_row(feature_names, user_inputs):
    """
    Creates one unscaled prediction row using the exact training feature names.
    Features that are not controlled by the interface are filled with X_train medians.
    This prevents shape mismatch and keeps the prediction aligned with training.
    """
    defaults = load_training_defaults()
    row = pd.Series({f: float(defaults.get(f, 0.0)) for f in feature_names}, dtype="float64")

    distance = user_inputs["flight_distance_km"]
    hours = user_inputs["flight_hours"]
    ticket_price = user_inputs["ticket_price"]
    passengers = user_inputs["passengers"]
    capacity = user_inputs["aircraft_capacity"]
    load_factor = passengers / max(capacity, 1)
    price_per_km = ticket_price / max(distance, 1)
    delay_flag = 1 if user_inputs["delay_minutes"] > 30 else 0
    is_weekend = 1 if user_inputs["departure_day"] in ["Saturday", "Sunday"] else 0

    # Direct numeric features — supports common naming variations safely.
    _set_if_exists(row, ["Flight_Distance_km", "flight_distance_km", "Distance_km", "Distance"], distance)
    _set_if_exists(row, ["Flight_Hours", "flight_hours"], hours)
    _set_if_exists(row, ["Avg_Ticket_Price", "Average_Ticket_Price", "Ticket_Price", "ticket_price"], ticket_price)
    _set_if_exists(row, ["Passengers", "passengers"], passengers)
    _set_if_exists(row, ["Aircraft_Capacity", "aircraft_capacity", "Capacity"], capacity)
    _set_if_exists(row, ["Load_Factor", "load_factor"], load_factor)
    _set_if_exists(row, ["price_per_km", "Price_per_km", "Price_Per_KM"], price_per_km)
    _set_if_exists(row, ["Delay_Minutes", "delay_minutes"], user_inputs["delay_minutes"])
    _set_if_exists(row, ["delay_flag", "Delay_Flag"], delay_flag)
    _set_if_exists(row, ["Competition_Index", "competition_index"], user_inputs["competition_index"])
    _set_if_exists(row, ["Aircraft_Age_Years", "aircraft_age_years"], user_inputs["aircraft_age_years"])
    _set_if_exists(row, ["flight_month", "Flight_Month", "Month"], user_inputs["flight_month"])
    _set_if_exists(row, ["is_weekend", "Is_Weekend"], is_weekend)

    # Ordinal features based on preprocessing description.
    season_ord = {"Off-Peak": 0, "Shoulder": 1, "Peak": 2}[user_inputs["season"]]
    demand_ord = {"Low": 0, "Medium": 1, "High": 2}[user_inputs["demand_level"]]
    _set_if_exists(row, ["Season_Ordinal", "season_ordinal"], season_ord)
    _set_if_exists(row, ["Demand_Level_Ordinal", "demand_level_ordinal"], demand_ord)

    # Aircraft body style.
    _set_if_exists(row, ["is_narrow_body", "Is_Narrow_Body"], 1 if user_inputs["aircraft_type"] == "Narrow-body" else 0)

    # One-hot encoded features. Only sets columns that exist in feature_names.
    one_hot_values = {
        f"Season_{user_inputs['season']}": 1,
        f"Demand_Level_{user_inputs['demand_level']}": 1,
        f"Route_Category_{user_inputs['route_category']}": 1,
        f"Alliance_{user_inputs['alliance']}": 1,
        f"Region_{user_inputs['region']}": 1,
        f"Aircraft_Type_{user_inputs['aircraft_type']}": 1,
    }
    for col, val in one_hot_values.items():
        if col in row.index:
            row[col] = val

    return pd.DataFrame([row], columns=feature_names)


def predict_with_trained_model(model_name, user_inputs, feature_names):
    model, model_path = load_trained_model(model_name)
    scaler = load_scaler()

    if model is None:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not feature_names:
        raise ValueError("feature_names.json is missing or empty.")

    X_raw = build_prediction_row(feature_names, user_inputs)
    X_model = scaler.transform(X_raw) if scaler is not None else X_raw.values

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_model)[0][1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X_model)[0])
        probability = float(1 / (1 + np.exp(-score)))
    else:
        pred = int(model.predict(X_model)[0])
        probability = float(pred)

    prediction = int(probability >= user_inputs["threshold"])
    return {
        "prediction": prediction,
        "label": "Profitable" if prediction == 1 else "Loss-making",
        "probability": probability,
        "threshold": user_inputs["threshold"],
        "model_path": str(model_path),
        "input_row": X_raw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Project Overview
# ═══════════════════════════════════════════════════════════════════════════
if page == "Project Overview":
    st.title("Airline Route Profitability Prediction")
    st.subheader("COM3032/COMM074 — Group Project Overview")

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
        ("M4", "KNN+PCA", "#E67E22", "PCA dimensionality reduction + distance-based voting. Permutation importance."),
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
    - **M4 (KNN)** — gini/entropy, no class balancing, deep/unconstrained growth
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
                "Multi-Layer Perceptron": "MLP",
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

        def axis_label(row):
            m = row["Member"].replace("Member ", "M")
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
            labels2 = [f"{row['Member']}\n{row['Model'].split(' ')[0]}"
                       for _, row in delta_df.iterrows()]
            ax2.set_xticks(x2)
            ax2.set_xticklabels(labels2, fontsize=9)
            ax2.set_ylabel("F1 Delta (Primary – Member DT)")
            ax2.set_title("F1 Improvement over Each Member's Decision Tree", fontweight="bold")
            ax2.grid(True, alpha=0.2, axis="y")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()


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
            st.subheader(f"{member} — {primary_row['Model']}")

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
                ax.set_title(f"{label} — {member}", fontweight="bold")
                ax.set_ylim(min(values) * 0.98, 1.0)
                ax.grid(True, alpha=0.2, axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Show saved figures if present
        st.divider()
        st.markdown("### Saved Visualisations")

        # Map member to figure prefix
        suffix_map = {"Member 1": "m1", "Member 2": "m2",
                      "Member 3": "m3", "Member 4": "m4", "Member 5": "m5"}
        suffix = suffix_map.get(member, "")
        model_prefix_map = {
            "Member 1": "rf", "Member 2": "xgb",
            "Member 3": "svm", "Member 4": "knn", "Member 5": "mlp"
        }
        mp = model_prefix_map.get(member, "")

        possible_figs = [
            f"{mp}_tuned_learning_curve_{suffix}.png",
            f"{mp}_feature_importance_{suffix}.png",
            f"perm_importance_{suffix}.png",
            f"{mp}_loss_curve_{suffix}.png",
            f"{mp}_cm_roc_test_{suffix}.png",
            f"dt_cm_roc_test_{suffix}.png",
            f"dt_tree_viz_{suffix}.png",
        ]

        found = [(f, m_dir / f) for f in possible_figs if (m_dir / f).exists()]
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
    | Real-time API endpoint | MLP or KNN (post-training) | Fast inference |
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
# PAGE 6 — Profitability Predictor
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Profitability Predictor":
    st.title("Profitability Prediction Interface")
    st.subheader("Real trained-model route profitability prediction")

    st.success(
        "This page now loads the saved trained model files from each member folder and uses "
        "processed_data/scaler.pkl + feature_names.json to make real predictions."
    )

    if not features:
        st.error("feature_names.json was not found or is empty. Please check the processed_data folder.")
        st.stop()

    available_models = [name for name, path in MODEL_PATHS.items() if path.exists()]
    missing_models = [name for name, path in MODEL_PATHS.items() if not path.exists()]

    if not available_models:
        st.error("No trained model .pkl files were found. Please check the Member folders.")
        st.stop()

    with st.expander("Model artefact status", expanded=False):
        status_df = pd.DataFrame({
            "Model": list(MODEL_PATHS.keys()),
            "File Path": [str(p) for p in MODEL_PATHS.values()],
            "Found": [p.exists() for p in MODEL_PATHS.values()],
        })
        st.dataframe(status_df, use_container_width=True)

    st.markdown("### 1. Select trained model")
    selected_model = st.selectbox(
        "Choose model file for prediction",
        available_models,
        index=0,
        help="This loads the actual saved .pkl model from your project folders."
    )

    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability above this value is classified as Profitable."
    )

    st.markdown("### 2. Enter route details")
    col1, col2, col3 = st.columns(3)

    with col1:
        flight_distance_km = st.number_input("Flight Distance (km)", min_value=100.0, max_value=20000.0, value=1200.0, step=50.0)
        flight_hours = st.number_input("Flight Hours", min_value=0.5, max_value=20.0, value=2.5, step=0.5)
        ticket_price = st.number_input("Average Ticket Price (£)", min_value=20.0, max_value=3000.0, value=180.0, step=10.0)
        passengers = st.number_input("Expected Passengers", min_value=1.0, max_value=700.0, value=150.0, step=5.0)

    with col2:
        aircraft_capacity = st.number_input("Aircraft Capacity", min_value=20.0, max_value=800.0, value=180.0, step=5.0)
        aircraft_age_years = st.number_input("Aircraft Age (years)", min_value=0.0, max_value=40.0, value=8.0, step=1.0)
        competition_index = st.slider("Competition Index", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        delay_minutes = st.number_input("Expected / Historical Delay Minutes", min_value=0.0, max_value=300.0, value=20.0, step=5.0)

    with col3:
        season = st.selectbox("Season", ["Off-Peak", "Shoulder", "Peak"], index=1)
        demand_level = st.selectbox("Demand Level", ["Low", "Medium", "High"], index=1)
        route_category = st.selectbox("Route Category", ["Domestic", "Regional", "International", "Long-haul"], index=2)
        alliance = st.selectbox("Alliance", ["None", "Star Alliance", "Oneworld", "SkyTeam"], index=0)
        region = st.selectbox("Region", ["Europe", "Asia", "North America", "Middle East", "Africa", "Oceania", "South America"], index=0)
        aircraft_type = st.selectbox("Aircraft Type", ["Narrow-body", "Wide-body", "Regional Jet"], index=0)
        flight_month = st.slider("Flight Month", min_value=1, max_value=12, value=7, step=1)
        departure_day = st.selectbox("Departure Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)

    user_inputs = {
        "flight_distance_km": flight_distance_km,
        "flight_hours": flight_hours,
        "ticket_price": ticket_price,
        "passengers": passengers,
        "aircraft_capacity": aircraft_capacity,
        "aircraft_age_years": aircraft_age_years,
        "competition_index": competition_index,
        "delay_minutes": delay_minutes,
        "season": season,
        "demand_level": demand_level,
        "route_category": route_category,
        "alliance": alliance,
        "region": region,
        "aircraft_type": aircraft_type,
        "flight_month": flight_month,
        "departure_day": departure_day,
        "threshold": threshold,
    }

    st.divider()

    if st.button("Predict Profitability", type="primary", use_container_width=True):
        try:
            result = predict_with_trained_model(selected_model, user_inputs, features)
            prob = result["probability"]
            label = result["label"]

            if label == "Profitable":
                st.success(f"Prediction: {label}")
            else:
                st.error(f"Prediction: {label}")

            st.markdown("### Prediction Result")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Selected Model", selected_model)
            r2.metric("Profitability Probability", f"{prob:.2%}")
            r3.metric("Decision Threshold", f"{threshold:.2f}")
            r4.metric("Predicted Class", label)

            load_factor = passengers / max(aircraft_capacity, 1)
            price_per_km = ticket_price / max(flight_distance_km, 1)
            estimated_revenue = ticket_price * passengers

            st.markdown("### Route Input Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Load Factor", f"{load_factor:.2%}")
            s2.metric("Price per km", f"£{price_per_km:.2f}")
            s3.metric("Estimated Revenue", f"£{estimated_revenue:,.0f}")
            s4.metric("Competition Index", f"{competition_index:.2f}")

            st.markdown("### Model File Used")
            st.code(result["model_path"])

            with st.expander("View exact feature row sent to the model", expanded=False):
                st.dataframe(result["input_row"].T.rename(columns={0: "Input Value"}), use_container_width=True)

            st.caption(
                "This prediction uses the saved trained model artefact. Features not entered manually are filled using "
                "training-set median values from processed_data/X_train.csv to keep the model input shape correct."
            )

        except Exception as e:
            st.error("Prediction failed. Please check that the selected model, scaler.pkl, feature_names.json, and X_train.csv match the training pipeline.")
            st.exception(e)
