"""
STEP 2: After training, run this file to launch the app.
Command: streamlit run app.py
"""

import streamlit as st
import pickle
import numpy as np

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ── Load model & encoders ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    import os
    if not os.path.exists("model.pkl"):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        df = pd.read_csv("churn_data.csv").drop(columns=["customerID"])
        text_cols = [c for c in df.columns if str(df[c].dtype) in ("object","str","string")]
        les = {}
        for col in text_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            les[col] = le
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        X = df.drop(columns=["Churn"]).astype(float)
        y = df["Churn"].astype(int)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        with open("model.pkl","wb") as f: pickle.dump(clf, f)
        with open("encoders.pkl","wb") as f: pickle.dump(les, f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_artifacts()

# ── Helper: encode a value using saved encoder ────────────
def encode(col, val):
    return int(encoders[col].transform([val])[0])

# ── UI ────────────────────────────────────────────────────
st.title("📡 Customer Churn Predictor")
st.markdown("Fill in the customer details below and click **Predict** to see if they are likely to churn.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender",          ["Male", "Female"])
    senior         = st.selectbox("Senior Citizen",  ["No", "Yes"])
    partner        = st.selectbox("Has Partner",      ["Yes", "No"])
    dependents     = st.selectbox("Has Dependents",   ["Yes", "No"])
    tenure         = st.slider("Tenure (months)",     0, 72, 12)
    phone_service  = st.selectbox("Phone Service",    ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines",   ["Yes", "No", "No phone service"])

with col2:
    internet       = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_sec     = st.selectbox("Online Security",  ["Yes", "No", "No internet service"])
    online_backup  = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
    device_prot    = st.selectbox("Device Protection",["Yes", "No", "No internet service"])
    tech_support   = st.selectbox("Tech Support",     ["Yes", "No", "No internet service"])
    streaming_tv   = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
    streaming_mov  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.divider()

col3, col4 = st.columns(2)
with col3:
    contract       = st.selectbox("Contract Type",   ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing",["Yes", "No"])
with col4:
    payment        = st.selectbox("Payment Method",  [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly        = st.number_input("Monthly Charges ($)",  0.0, 200.0, 65.0, step=0.01)
    total          = st.number_input("Total Charges ($)",    0.0, 9000.0, 1200.0, step=0.01)

st.divider()

# ── Predict button ────────────────────────────────────────
if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):

    senior_val = 1 if senior == "Yes" else 0

    features = np.array([[
        encode("gender",          gender),
        senior_val,
        encode("Partner",         partner),
        encode("Dependents",      dependents),
        tenure,
        encode("PhoneService",    phone_service),
        encode("MultipleLines",   multiple_lines),
        encode("InternetService", internet),
        encode("OnlineSecurity",  online_sec),
        encode("OnlineBackup",    online_backup),
        encode("DeviceProtection",device_prot),
        encode("TechSupport",     tech_support),
        encode("StreamingTV",     streaming_tv),
        encode("StreamingMovies", streaming_mov),
        encode("Contract",        contract),
        encode("PaperlessBilling",paperless),
        encode("PaymentMethod",   payment),
        monthly,
        total,
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    churn_label = encoders["Churn"].inverse_transform([prediction])[0]
    churn_prob  = probability[1] * 100   # probability of churn class

    st.divider()
    if churn_label == "Yes":
        st.error(f"⚠️  **This customer is likely to CHURN**")
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
        st.markdown("**Recommendation:** Consider offering a discount or a loyalty plan.")
    else:
        st.success(f"✅  **This customer is NOT likely to churn**")
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
        st.markdown("**Recommendation:** Customer appears satisfied — maintain engagement.")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ using Streamlit & Random Forest · Customer Churn Prediction")
