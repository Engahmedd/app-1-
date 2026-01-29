from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Load & Prepare Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")

    if 'customerid' in df.columns:
        df.drop(columns=['customerid'], inplace=True)

    df_dashboard = df.copy()

    df_ml = df.copy()
    df_ml['churn'] = df_ml['churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = [
        'gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'paperlessbilling',
        'internetservice', 'contract', 'paymentmethod'
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        encoders[col] = le

    scaler = StandardScaler()
    num_cols = ["tenure", "monthlycharges", "totalcharges"]
    df_ml[num_cols] = scaler.fit_transform(df_ml[num_cols])

    X = df_ml.drop('churn', axis=1)
    y = df_ml['churn']

    return df_dashboard, X, y, encoders, scaler


@st.cache_resource
def load_model():
    # الموديل الحالي بيرجع 0/1 بعد التعديل
    return joblib.load("logistic_regression_model.pkl")

# =========================
# Analysis Page
# =========================
def analysis_page(df):
    st.title("📊 Telecom Churn Analysis Dashboard")

    col1, col2 = st.columns(2)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{(df['churn'] == 'Yes').mean() * 100:.2f}%")

    st.divider()

    st.subheader("Payment Method Distribution")
    payment_counts = df['paymentmethod'].value_counts().reset_index()
    payment_counts.columns = ['Payment Method', 'Count']
    st.plotly_chart(
        px.bar(payment_counts, x='Payment Method', y='Count'),
        width='stretch'
    )

    st.subheader("Contract Types")
    st.plotly_chart(
        px.pie(df, names='contract'),
        width='stretch'
    )

    st.subheader("Churn Distribution")
    st.plotly_chart(
        px.pie(df, names='churn'),
        width='stretch'
    )

    st.subheader("Contract vs Churn")
    st.plotly_chart(
        px.histogram(df, x='contract', color='churn', barmode='group'),
        width='stretch'
    )

    st.subheader("Monthly Charges Distribution")
    st.plotly_chart(
        px.histogram(df, x='monthlycharges', nbins=60),
        width='stretch'
    )

    cols = [
        'techsupport', 'streamingtv', 'streamingmovies',
        'paperlessbilling', 'paymentmethod'
    ]

    for col in cols:
        st.subheader(f"{col} vs Churn")
        st.plotly_chart(
            px.histogram(df, x=col, color='churn', barmode='group'),
            width='stretch'
        )

# =========================
# Prediction Page
# =========================
def prediction_page():
    st.title("🤖 Customer Churn Prediction")

    gender = st.selectbox("Gender", encoders['gender'].classes_)
    seniorcitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", encoders['partner'].classes_)
    dependents = st.selectbox("Dependents", encoders['dependents'].classes_)
    tenure = st.number_input("Tenure (Months)", 0, 100, 10)
    phoneservice = st.selectbox("Phone Service", encoders['phoneservice'].classes_)
    multiplelines = st.selectbox("Multiple Lines", encoders['multiplelines'].classes_)
    onlinesecurity = st.selectbox("Online Security", encoders['onlinesecurity'].classes_)
    onlinebackup = st.selectbox("Online Backup", encoders['onlinebackup'].classes_)
    deviceprotection = st.selectbox("Device Protection", encoders['deviceprotection'].classes_)
    techsupport = st.selectbox("Tech Support", encoders['techsupport'].classes_)
    streamingtv = st.selectbox("Streaming TV", encoders['streamingtv'].classes_)
    streamingmovies = st.selectbox("Streaming Movies", encoders['streamingmovies'].classes_)
    paperlessbilling = st.selectbox("Paperless Billing", encoders['paperlessbilling'].classes_)
    monthlycharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    totalcharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
    internetservice = st.selectbox("Internet Service", encoders['internetservice'].classes_)
    contract = st.selectbox("Contract", encoders['contract'].classes_)
    paymentmethod = st.selectbox("Payment Method", encoders['paymentmethod'].classes_)

    input_data = {
        'gender': encoders['gender'].transform([gender])[0],
        'seniorcitizen': 1 if seniorcitizen == "Yes" else 0,
        'partner': encoders['partner'].transform([partner])[0],
        'dependents': encoders['dependents'].transform([dependents])[0],
        'tenure': tenure,
        'phoneservice': encoders['phoneservice'].transform([phoneservice])[0],
        'multiplelines': encoders['multiplelines'].transform([multiplelines])[0],
        'onlinesecurity': encoders['onlinesecurity'].transform([onlinesecurity])[0],
        'onlinebackup': encoders['onlinebackup'].transform([onlinebackup])[0],
        'deviceprotection': encoders['deviceprotection'].transform([deviceprotection])[0],
        'techsupport': encoders['techsupport'].transform([techsupport])[0],
        'streamingtv': encoders['streamingtv'].transform([streamingtv])[0],
        'streamingmovies': encoders['streamingmovies'].transform([streamingmovies])[0],
        'paperlessbilling': encoders['paperlessbilling'].transform([paperlessbilling])[0],
        'monthlycharges': monthlycharges,
        'totalcharges': totalcharges,
        'internetservice': encoders['internetservice'].transform([internetservice])[0],
        'contract': encoders['contract'].transform([contract])[0],
        'paymentmethod': encoders['paymentmethod'].transform([paymentmethod])[0],
    }

    df_input = pd.DataFrame([input_data])
    df_input[['tenure', 'monthlycharges', 'totalcharges']] = scaler.transform(
        df_input[['tenure', 'monthlycharges', 'totalcharges']]
    )

    if st.button("Predict Churn"):
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

        # لو الموديل بيرجع 0/1
        if pred == 1:
            st.error(f"❌ Customer is likely to churn (Probability: {proba:.2%})")
        else:
            st.success(f"✅ Customer is not likely to churn (Probability: {proba:.2%})")

    # Model performance
    y_pred = model.predict(X)
    y_pred = [1 if val == 'Yes' else 0 for val in y_pred]  # حل مشكلة النوع
    acc = accuracy_score(y, y_pred)
    st.subheader("📈 Model Performance")
    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# =========================
# Main App
# =========================
st.set_page_config(
    page_title="Telecom Churn AI",
    page_icon="📊",
    layout="wide"
)

st.title("📡 Telecom Customer Churn AI System")
st.markdown(
    """
    This system helps analyze customer behavior  
    and predict whether a customer is likely to churn.
    """
)

tab1, tab2 = st.tabs(["📊 Analysis Dashboard", "🤖 Churn Prediction"])

df_dashboard, X, y, encoders, scaler = load_data()
model = load_model()

with tab1:
    analysis_page(df_dashboard)

with tab2:
    prediction_page()
