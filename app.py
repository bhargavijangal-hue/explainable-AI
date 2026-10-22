import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ClinExplain AI | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling ---------------------------------------
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #101a34; }
    [data-testid="stSidebar"] * { color: #f4f7ff !important; }
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(120deg,#122348,#2563eb);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #e4ecff; margin: .45rem 0 0 0; }
    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1rem; box-shadow: 0 2px 10px rgba(15,23,42,.04);
    }
    .muted { color: #64748b; font-size: .88rem; }
    .warning-box {
        padding: .9rem 1rem; border-radius: 12px; background: #fff7ed;
        border: 1px solid #fed7aa; color: #9a3412;
    }
    .success-box {
        padding: .9rem 1rem; border-radius: 12px; background: #ecfdf5;
        border: 1px solid #a7f3d0; color: #065f46;
    }
    div[data-testid="stMetric"] {
        background: white; padding: 12px 15px; border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Model -----------------------------------------
@st.cache_resource
def train_model():
    """Train a demo model on sklearn's built-in breast-cancer dataset."""
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    # sklearn target: 0=malignant, 1=benign. Present the clinically clearer labels.
    y = pd.Series(dataset.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=180, random_state=42, class_weight="balanced"
        )),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, probs[:, 1]),
        "confusion": confusion_matrix(y_test, pred),
        "report": classification_report(
            y_test, pred, target_names=["Malignant", "Benign"], output_dict=True
        ),
    }
    return dataset, X, y, X_train, X_test, y_train, y_test, pipeline, metrics

dataset, X, y, X_train, X_test, y_train, y_test, model, metrics = train_model()
FEATURES = list(X.columns)
CLASS_NAMES = ["Malignant (higher concern)", "Benign (lower concern)"]

def pretty_feature(name: str) -> str:
    return name.replace(" ", " ").title()

def local_lime(values: np.ndarray, num_features: int = 8):
    explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=FEATURES,
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=17,
    )
    explanation = explainer.explain_instance(
        data_row=values,
        predict_fn=model.predict_proba,
        num_features=num_features,
        top_labels=2,
    )
    label = int(model.predict(values.reshape(1, -1))[0])
    pairs = explanation.as_list(label=label)
    return label, explanation, pairs

def make_counterfactual(values: np.ndarray, target_class: int, max_changes: int = 4):
    """
    Simple, transparent demo counterfactual search.
    Varies one feature at a time around observed training quantiles and selects
    realistic candidates that change the model prediction. This is not DiCE.
    """
    base = values.copy()
    base_pred = int(model.predict(base.reshape(1, -1))[0])
    if base_pred == target_class:
        return [], base_pred
    candidates = []
    quantiles = X_train.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    for i, feature in enumerate(FEATURES):
        original = base[i]
        for q in quantiles.index:
            trial = base.copy()
            trial[i] = float(quantiles.loc[q, feature])
            pred = int(model.predict(trial.reshape(1, -1))[0])
            if pred == target_class:
                delta = abs(trial[i] - original) / (X_train[feature].std() + 1e-9)
                candidates.append({
                    "feature": feature,
                    "original": original,
                    "new": trial[i],
                    "change": trial[i] - original,
                    "normalized_change": delta,
                    "prediction": pred,
                })
    candidates.sort(key=lambda item: item["normalized_change"])
    selected, seen = [], set()
    for item in candidates:
        if item["feature"] not in seen:
            selected.append(item)
            seen.add(item["feature"])
        if len(selected) >= max_changes:
            break
    return selected, base_pred

def show_metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

# ----------------------------- Sidebar ---------------------------------------
with st.sidebar:
    st.markdown("## 🩺 ClinExplain AI")
    st.caption("Explainable clinical decision support")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Diagnosis & Explanation", "Model Evaluation", "About Project"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### Demo dataset")
    st.write("Breast Cancer Wisconsin (Diagnostic)")
    st.caption("Built-in scikit-learn educational dataset")
    st.markdown("---")
    st.caption("Prototype • Not for clinical diagnosis")

# ----------------------------- Header ----------------------------------------
st.markdown("""
<div class="hero">
  <h1>ClinExplain AI</h1>
  <p>Transparent machine-learning predictions with local LIME explanations and
  what-if counterfactual examples.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
<b>Research / education prototype:</b> This app is not a medical device and must
not be used to diagnose, treat, or rule out disease. It uses a public demonstration
dataset, not a patient's validated clinical record. Always defer to qualified
healthcare professionals.
</div>
""", unsafe_allow_html=True)

# ----------------------------- Pages -----------------------------------------
if page == "Dashboard":
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: show_metric("Model accuracy", f"{metrics['accuracy']:.1%}", "Held-out test split")
    with c2: show_metric("ROC-AUC", f"{metrics['auc']:.3f}", "Held-out test split")
    with c3: show_metric("Training samples", f"{len(X_train):,}")
    with c4: show_metric("Input features", f"{X.shape[1]}")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### What this prototype does")
        st.write(
            "1. Uses a trained machine-learning classifier on a public dataset.\n"
            "2. Predicts one of two dataset labels for an example input.\n"
            "3. Uses LIME to explain the local prediction.\n"
            "4. Searches for a small set of hypothetical input changes that can "
            "flip the model output."
        )
        st.markdown("#### Suggested workflow")
        st.info("Open **Diagnosis & Explanation**, choose an example record or enter feature values, then generate the explanation.")
    with right:
        st.markdown("#### Dataset label balance")
        counts = y.map({0: "Malignant", 1: "Benign"}).value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.3))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Number of records")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Important limitations")
    st.write(
        "- Test-set metrics are not evidence of clinical safety or generalization to real hospitals.\n"
        "- LIME is a local approximation and may vary with sampling settings.\n"
        "- Counterfactuals are hypothetical model inputs, not treatment recommendations.\n"
        "- This demo does not store patient data or connect to hospital systems."
    )

elif page == "Diagnosis & Explanation":
    st.subheader("Single-record prediction")
    st.write("Select a demo record or adjust the feature values. The dataset contains breast-tumour measurements, not general patient vital signs.")

    options = {
        "Example 1 — record predicted benign": int(np.where(model.predict(X.to_numpy()) == 1)[0][0]),
        "Example 2 — record predicted malignant": int(np.where(model.predict(X.to_numpy()) == 0)[0][0]),
        "Random test record": int(X_test.index[0]),
    }
    selected_label = st.selectbox("Starting example", list(options.keys()))
    record_idx = options[selected_label]
    start_values = X.loc[record_idx].to_numpy(dtype=float)

    with st.expander("Edit input features", expanded=False):
        st.caption("Values are measurement features from the public dataset. Keep values within plausible dataset ranges.")
        edited_values = start_values.copy()
        columns = st.columns(3)
        for i, feature in enumerate(FEATURES):
            with columns[i % 3]:
                min_v = float(X[feature].quantile(0.01))
                max_v = float(X[feature].quantile(0.99))
                if max_v <= min_v:
                    max_v = min_v + 1.0
                edited_values[i] = st.number_input(
                    pretty_feature(feature),
                    min_value=min_v,
                    max_value=max_v,
                    value=float(np.clip(start_values[i], min_v, max_v)),
                    key=f"feature_{i}",
                    format="%.4f",
                )

    if st.button("Generate prediction & explanations", type="primary", use_container_width=True):
        pred_proba = model.predict_proba(edited_values.reshape(1, -1))[0]
        pred = int(np.argmax(pred_proba))
        label_text = "Benign (lower concern in this dataset)" if pred == 1 else "Malignant (higher concern in this dataset)"
        confidence = float(pred_proba[pred])

        st.markdown("### Prediction")
        p1, p2, p3 = st.columns(3)
        with p1: st.metric("Model output", label_text)
        with p2: st.metric("Probability for selected class", f"{confidence:.1%}")
        with p3: st.metric("Alternative class probability", f"{pred_proba[1-pred]:.1%}")

        st.markdown("### LIME local explanation")
        try:
            lime_label, explanation, pairs = local_lime(edited_values, num_features=10)
            lime_df = pd.DataFrame(pairs, columns=["Feature condition", "Local contribution"])
            lime_df["Direction"] = np.where(
                lime_df["Local contribution"] >= 0, "Supports selected class", "Opposes selected class"
            )
            st.dataframe(lime_df, use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            plot_df = lime_df.sort_values("Local contribution")
            ax.barh(plot_df["Feature condition"], plot_df["Local contribution"])
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("LIME local contribution (model-local; not causal)")
            ax.set_ylabel("")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Could not create the LIME explanation: {exc}")

        st.markdown("### Counterfactual what-if examples")
        target = 1 - pred
        st.write(f"The search below looks for hypothetical input changes that make the model predict the alternative class: **{CLASS_NAMES[target]}**.")
        counterfactuals, original_pred = make_counterfactual(edited_values, target_class=target)
        if counterfactuals:
            cf_df = pd.DataFrame(counterfactuals)
            cf_df["feature"] = cf_df["feature"].map(pretty_feature)
            cf_df = cf_df.rename(columns={
                "feature": "Feature",
                "original": "Original value",
                "new": "Hypothetical value",
                "change": "Difference",
                "normalized_change": "Scaled change size",
                "prediction": "Target class code",
            })
            st.dataframe(cf_df, use_container_width=True, hide_index=True)
            st.caption("These are model-flipping examples found by a simple quantile search. They are not guaranteed to be clinically plausible, actionable, causal, or minimal across multiple features.")
        else:
            st.warning("No single-feature counterfactual was found in the searched quantiles. This does not mean the prediction cannot change; a multi-feature search or different method may be needed.")

        with st.expander("Show entered feature values"):
            st.dataframe(pd.DataFrame({"Feature": FEATURES, "Value": edited_values}), use_container_width=True, hide_index=True)

        st.download_button(
            "Download explanation as CSV",
            data=pd.DataFrame({"Feature": FEATURES, "Value": edited_values}).to_csv(index=False).encode("utf-8"),
            file_name="clinical_explanation_input.csv",
            mime="text/csv",
        )

elif page == "Model Evaluation":
    st.subheader("Model evaluation")
    st.write("Metrics below use a held-out split from the built-in demonstration dataset. They are for coursework demonstration only.")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("ROC-AUC", f"{metrics['auc']:.3f}")
    with c2:
        st.markdown("#### Confusion matrix")
        cm = metrics["confusion"]
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        im = ax.imshow(cm)
        ax.set_xticks([0, 1], labels=["Malignant", "Benign"])
        ax.set_yticks([0, 1], labels=["Malignant", "Benign"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    report = pd.DataFrame(metrics["report"]).T
    st.markdown("#### Classification report")
    st.dataframe(report.round(3), use_container_width=True)
    st.caption("The dataset's labels and performance may not represent real-world clinical populations.")

elif page == "About Project":
    st.subheader("About this project")
    st.markdown("""
**Project title:** Counterfactual Explanation Generation for Decision Support via LIME Local Surrogate Model Explanations, Applied to Clinical Decision Support Systems for Diagnosis Explanation.

**Problem:** Predictive models can be difficult to interpret, which can make it harder for users to assess whether a prediction is sensible.

**Approach:**
- Train a Random Forest classifier on a public demonstration dataset.
- Use LIME to provide a local, per-record explanation.
- Search feature quantiles for hypothetical inputs that flip the predicted class.
- Display the results in a web dashboard.

**Technology stack:** Python, Streamlit, Pandas, NumPy, Matplotlib, scikit-learn, LIME.

**Future improvements:**
- Use a clinically reviewed, appropriately licensed dataset.
- Add DiCE or another dedicated counterfactual method with feasibility constraints.
- Add calibration, subgroup evaluation, and external validation.
- Add access control, audit logging, privacy controls, and a formal clinical safety review before any deployment.
- Integrate with healthcare systems only through approved, secure standards and governance.
""")
    st.markdown("#### Academic integrity & safe use")
    st.write("This is a working educational prototype, not a clinically validated diagnostic system. It should not be used for real patient-care decisions.")
