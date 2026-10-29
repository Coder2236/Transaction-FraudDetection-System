"""Streamlit fraud detection demo.

Run:  streamlit run app.py
Needs (in the same folder): fraud_model.json or fraud_model.joblib, and model_meta.json
Optional: sample_transactions.csv, results_table.csv, cost_table.csv,
          leakage_comparison.csv, tuning_results.csv, cv_summary.csv, figures/*.png
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud detector", layout="wide")


@st.cache_resource
def load_artifacts():
    with open("model_meta.json") as f:
        meta = json.load(f)
    if meta.get("model_format") == "xgb_json" or os.path.exists("fraud_model.json"):
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model("fraud_model.json")
    else:
        model = joblib.load("fraud_model.joblib")
    explainer = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
    except Exception:
        pass
    return model, meta, explainer


has_model = os.path.exists("fraud_model.json") or os.path.exists("fraud_model.joblib")
if not (has_model and os.path.exists("model_meta.json")):
    st.error("Model files not found. Put fraud_model.json (or fraud_model.joblib) and model_meta.json in the same folder as app.py.")
    st.stop()

model, meta, explainer = load_artifacts()


def model_feature_names(m):
    """The feature list the model was actually trained on (more reliable than model_meta.json)."""
    for getter in (lambda: m.get_booster().feature_names, lambda: m.feature_names_in_):
        try:
            names = getter()
            if names:
                return list(names)
        except Exception:
            pass
    return None


FEATURES = model_feature_names(model) or meta["features"]
FULL_MODE = "newbalanceOrig" in FEATURES   # model needs balances after the transaction
RISKY = meta["risky_types"]
ALL_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as the notebook. Uses only information known BEFORE the transaction."""
    amt = df["amount"].astype(float).values
    old_o = df["oldbalanceOrg"].astype(float).values
    old_d = df["oldbalanceDest"].astype(float).values
    cols = {
        "amount": amt,
        "hour": df["hour"].astype(int).values,
        "is_transfer": (df["type"] == "TRANSFER").astype(int).values,
        "oldbalanceOrg": old_o,
        "oldbalanceDest": old_d,
        "amt_to_orig_bal": amt / (old_o + 1),
        "amt_to_dest_bal": amt / (old_d + 1),
        "drains_account": ((old_o > 0) & (amt >= old_o - 0.01)).astype(int),
        "orig_empty": (old_o == 0).astype(int),
        "dest_empty": (old_d == 0).astype(int),
    }
    if "newbalanceOrig" in df.columns and "newbalanceDest" in df.columns:
        new_o = df["newbalanceOrig"].astype(float).values
        new_d = df["newbalanceDest"].astype(float).values
        cols["newbalanceOrig"] = new_o
        cols["newbalanceDest"] = new_d
        cols["errOrig"] = new_o + amt - old_o
        cols["errDest"] = old_d + amt - new_d
    return pd.DataFrame(cols)[FEATURES]


def score(df: pd.DataFrame) -> np.ndarray:
    """Fraud probability per row. Types that never contain fraud get 0."""
    prob = np.zeros(len(df))
    risky = df["type"].isin(RISKY).values
    if risky.any():
        prob[risky] = model.predict_proba(make_features(df[risky]))[:, 1]
    return prob


def shap_values(X: pd.DataFrame) -> np.ndarray:
    sv = explainer(X)
    v = sv.values
    return v[:, :, 1] if v.ndim == 3 else v


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
default_thr = float(np.clip(meta.get("threshold_cost", 0.5), 0.01, 0.99))
threshold = st.sidebar.slider("Fraud threshold", 0.01, 0.99, round(default_thr, 2), 0.01)
st.sidebar.caption(
    "A transaction is flagged when its fraud probability is at or above this value. "
    "Lower catches more fraud but raises false alarms. "
    f"Default is the cost-optimal threshold ({default_thr:.2f})."
)
st.sidebar.markdown(f"**Model:** {meta['model_name']}")
if FEATURES != meta["features"]:
    st.sidebar.warning(
        "The model file and model_meta.json come from different runs, so the app is using the model's own "
        "feature list. To fix it, delete the old model files and unzip the latest fraud_artifacts.zip again."
    )

try:
    import sklearn
    import xgboost
    if (sklearn.__version__ != meta.get("sklearn_version")) or (xgboost.__version__ != meta.get("xgboost_version")):
        st.sidebar.warning(
            f"Library versions differ from training (sklearn {meta.get('sklearn_version')}, "
            f"xgboost {meta.get('xgboost_version')}). Predictions may still work, but if you see errors, "
            "install those exact versions."
        )
except Exception:
    pass

st.title("Transaction fraud detector")
st.caption("Mobile-money fraud detection trained on the PaySim synthetic dataset. Demo project, not a production system.")
if FULL_MODE:
    st.info("This model uses all PaySim columns, including account balances after the transaction. "
            "Those are only known once a transaction is processed, so the scores here are higher than a "
            "pre-approval system could reach.")

tab1, tab2, tab3 = st.tabs(["Check one transaction", "Batch upload", "Model performance"])

# ---------------------------------------------------------------- tab 1
with tab1:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        txn_type = c1.selectbox("Transaction type", ALL_TYPES, index=1)
        amount = c2.number_input("Amount", min_value=0.0, value=250000.0, step=1000.0)
        hour = c3.slider("Hour of day (0-23)", 0, 23, 2)
        d1, d2 = st.columns(2)
        old_orig = d1.number_input("Sender balance before the transaction", min_value=0.0, value=300000.0, step=1000.0)
        old_dest = d2.number_input("Receiver balance before the transaction", min_value=0.0, value=0.0, step=1000.0)
        new_orig = new_dest = 0.0
        if FULL_MODE:
            e1, e2 = st.columns(2)
            new_orig = e1.number_input("Sender balance after the transaction", min_value=0.0, value=50000.0, step=1000.0)
            new_dest = e2.number_input("Receiver balance after the transaction", min_value=0.0, value=250000.0, step=1000.0)
        go = st.form_submit_button("Check transaction")

    if go:
        row = pd.DataFrame([{"type": txn_type, "amount": amount, "hour": hour,
                             "oldbalanceOrg": old_orig, "oldbalanceDest": old_dest,
                             "newbalanceOrig": new_orig, "newbalanceDest": new_dest}])
        prob = float(score(row)[0])
        left, right = st.columns([1, 2])
        with left:
            st.metric("Fraud probability", f"{prob:.1%}")
            if prob >= threshold:
                st.error("Flagged as likely fraud")
            else:
                st.success("Looks legitimate")
            if txn_type not in RISKY:
                st.info(f"{txn_type} transactions never contained fraud in the training data, "
                        "so they are treated as legitimate.")
        with right:
            if txn_type in RISKY and explainer is not None:
                X = make_features(row)
                vals = shap_values(X)[0]
                order = np.argsort(np.abs(vals))
                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.barh(np.array(FEATURES)[order], vals[order],
                        color=["#c0392b" if v > 0 else "#2e6da4" for v in vals[order]])
                ax.axvline(0, color="grey", lw=0.8)
                ax.set_xlabel("Effect on fraud score (SHAP)")
                ax.set_title("Why this score?")
                st.pyplot(fig)
                st.caption("Red bars push toward fraud, blue bars push toward legitimate. "
                           f"Inputs: type={txn_type}, amount={amount:,.0f}, hour={hour}, "
                           f"sender balance={old_orig:,.0f}, receiver balance={old_dest:,.0f}.")

# ---------------------------------------------------------------- tab 2
with tab2:
    need_txt = "**type**, **amount**, **oldbalanceOrg**, **oldbalanceDest**" + (
        ", **newbalanceOrig**, **newbalanceDest**" if FULL_MODE else "")
    st.write(f"Upload a CSV with columns {need_txt}, and either **step** or **hour**. "
             "An optional **isFraud** column (0/1) lets the app show precision and recall.")
    up = st.file_uploader("Transactions CSV", type="csv")
    use_sample = False
    if os.path.exists("sample_transactions.csv"):
        use_sample = st.button("Use the sample file instead")
    df_in = None
    if up is not None:
        df_in = pd.read_csv(up)
    elif use_sample or st.session_state.get("sample_loaded"):
        st.session_state["sample_loaded"] = True
        df_in = pd.read_csv("sample_transactions.csv")

    if df_in is not None:
        d = df_in.copy()
        if "hour" not in d.columns and "step" in d.columns:
            d["hour"] = d["step"] % 24
        required = ["type", "amount", "hour", "oldbalanceOrg", "oldbalanceDest"] + (
            ["newbalanceOrig", "newbalanceDest"] if FULL_MODE else [])
        missing = [c for c in required if c not in d.columns]
        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}. Add them (or `step` instead of `hour`) and upload again.")
        else:
            d["fraud_probability"] = score(d)
            d["verdict"] = np.where(d["fraud_probability"] >= threshold, "Fraud", "Legitimate")
            d = d.sort_values("fraud_probability", ascending=False).reset_index(drop=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Transactions", f"{len(d):,}")
            m2.metric("Flagged as fraud", f"{(d['verdict'] == 'Fraud').sum():,}")
            m3.metric("Flag rate", f"{(d['verdict'] == 'Fraud').mean():.1%}")

            if "isFraud" in d.columns:
                tp = int(((d["verdict"] == "Fraud") & (d["isFraud"] == 1)).sum())
                fp = int(((d["verdict"] == "Fraud") & (d["isFraud"] == 0)).sum())
                fn = int(((d["verdict"] == "Legitimate") & (d["isFraud"] == 1)).sum())
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                n1, n2, n3 = st.columns(3)
                n1.metric("Precision on this file", f"{prec:.1%}")
                n2.metric("Recall on this file", f"{rec:.1%}")
                n3.metric("Missed frauds", f"{fn:,}")

            st.dataframe(d, height=380)
            st.download_button("Download results (CSV)", d.to_csv(index=False).encode(),
                               "fraud_scored.csv", "text/csv")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write(f"Final model: **{meta['model_name']}**. Numbers are from the held-out test set.")
    if os.path.exists("results_table.csv"):
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv("results_table.csv"))
    if os.path.exists("cost_table.csv"):
        st.subheader(f"Cost-based threshold (missed fraud = {meta['cost_fn']:,}, false alarm = {meta['cost_fp']:,})")
        st.dataframe(pd.read_csv("cost_table.csv"))
    if os.path.exists("cv_summary.csv"):
        st.subheader("Cross-validation of the tuned XGBoost (training data)")
        st.dataframe(pd.read_csv("cv_summary.csv", index_col=0))
        st.caption("Stratified k-fold cross-validation: a small standard deviation means the score is stable across folds.")
    if os.path.exists("tuning_results.csv"):
        st.subheader("Hyperparameter search (top 5 of the combinations tried)")
        st.dataframe(pd.read_csv("tuning_results.csv").head(5))
    if os.path.exists("leakage_comparison.csv"):
        st.subheader("Feature comparison")
        st.dataframe(pd.read_csv("leakage_comparison.csv", index_col=0))
        st.caption("Basic and pre-transaction features use only information known before a transaction is approved. "
                   "The full PaySim columns also include balances after the transaction. PaySim cancels fraudulent "
                   "transactions, so those columns reveal the outcome and push scores close to perfect.")
    for img, cap in [("figures/pr_curves_and_confusion.png", "Precision-recall curves and confusion matrix"),
                     ("figures/shap_summary.png", "SHAP summary: which features drive predictions"),
                     ("figures/shap_waterfall_example.png", "SHAP explanation of one real fraud")]:
        if os.path.exists(img):
            st.image(img, caption=cap)
