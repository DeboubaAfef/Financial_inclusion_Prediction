import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# App title and styling
st.set_page_config(page_title="Financial Inclusion Predictor", layout="wide")
st.title("💰 Financial Inclusion Prediction App")
st.markdown("Predict whether an individual has a **bank account** based on demographic and socio-economic features.")

# Sidebar input method
st.sidebar.header("📥 Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# Define label encodings (used during training)
label_maps = {
    'country': {'Kenya': 0, 'Rwanda': 1, 'Tanzania': 2, 'Uganda': 3},
    'location_type': {'Rural': 0, 'Urban': 1},
    'cellphone_access': {'No': 0, 'Yes': 1},
    'gender_of_respondent': {'Female': 0, 'Male': 1},
    'relationship_with_head': {
        'Head of Household': 0, 'Spouse': 4, 'Child': 1, 'Parent': 3, 'Other relative': 2, 'Other non-relatives': 5
    },
    'marital_status': {
        'Married/Living together': 2, 'Single/Never Married': 4, 'Widowed': 5,
        'Divorced/Separated': 1, 'Don’t know': 0, 'Other': 3
    },
    'education_level': {
        'Secondary education': 4, 'No formal education': 1, 'Primary education': 3,
        'Tertiary education': 5, 'Vocational/Specialised training': 6, 'Other/Dont know/RTA': 2, '6': 0
    },
    'job_type': {
        'Self employed': 5, 'Government Dependent': 1, 'Formally employed Private': 0,
        'Informally employed': 3, 'Formally employed Government': 2,
        'Farming and Fishing': 4, 'Remittance Dependent': 6, 'Other Income': 7,
        'No Income': 8, 'Dont Know/Refuse to answer': 9
    }
}

# Feature names expected by the model
expected_features = [
    'country', 'year', 'location_type', 'cellphone_access', 'household_size',
    'age_of_respondent', 'gender_of_respondent', 'relationship_with_head',
    'marital_status', 'education_level', 'job_type'
]

# Helper function to encode a single row manually
def encode_input(data):
    for col in label_maps:
        if col in data.columns:
            data[col] = data[col].map(label_maps[col])
    return data

# 📥 Manual Input
if input_method == "Manual Input":
    st.subheader("✍️ Enter the person's information")

    with st.form("user_input_form"):
        country = st.selectbox("Country", list(label_maps['country'].keys()))
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2018)
        location_type = st.selectbox("Location Type", list(label_maps['location_type'].keys()))
        cellphone_access = st.selectbox("Cellphone Access", list(label_maps['cellphone_access'].keys()))
        household_size = st.number_input("Household Size", min_value=1, max_value=20, value=4)
        age = st.slider("Age of Respondent", 16, 100, 35)
        gender = st.selectbox("Gender", list(label_maps['gender_of_respondent'].keys()))
        relationship = st.selectbox("Relationship with Head", list(label_maps['relationship_with_head'].keys()))
        marital_status = st.selectbox("Marital Status", list(label_maps['marital_status'].keys()))
        education = st.selectbox("Education Level", list(label_maps['education_level'].keys()))
        job_type = st.selectbox("Job Type", list(label_maps['job_type'].keys()))

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame({
                'country': [country],
                'year': [year],
                'location_type': [location_type],
                'cellphone_access': [cellphone_access],
                'household_size': [household_size],
                'age_of_respondent': [age],
                'gender_of_respondent': [gender],
                'relationship_with_head': [relationship],
                'marital_status': [marital_status],
                'education_level': [education],
                'job_type': [job_type]
            })

            input_encoded = encode_input(input_data)
            prediction = model.predict(input_encoded)[0]
            label = "✅ Has Bank Account" if prediction == 1 else "❌ No Bank Account"
            st.success(f"Prediction: **{label}**")

# 📤 CSV Upload
elif input_method == "Upload CSV":
    st.subheader("📂 Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("📄 Preview of uploaded data:")
        st.dataframe(df_uploaded.head())

        # Ensure expected columns are present
        if all(col in df_uploaded.columns for col in expected_features):
            df_encoded = encode_input(df_uploaded.copy())
            predictions = model.predict(df_encoded)
            df_uploaded["prediction"] = np.where(predictions == 1, "Has Bank Account", "No Bank Account")

            st.success("✅ Prediction completed!")
            st.write(df_uploaded[["prediction"]].value_counts().rename("count"))

            st.subheader("📊 Results with Predictions")
            st.dataframe(df_uploaded)

            csv_download = df_uploaded.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", data=csv_download, file_name="predictions.csv", mime="text/csv")
        else:
            st.error("❌ Uploaded CSV is missing required columns.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and XGBoost - 2025")
