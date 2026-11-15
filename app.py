
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load saved model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = joblib.load("credit_risk_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, encoder, scaler


model, one, ss = load_artifacts()

# Exact feature order expected by the trained model
feature_columns = list(model.estimators_[0].feature_names_in_)

# Numerical columns used during scaling
num_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

# Missing-value medians from training data
missing_values = {
    "person_emp_length": 4.0,
    "loan_int_rate": 10.99
}

# Loan grade encoding (LabelEncoder alphabetical order)
grade_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

# Application title
st.title("🏦 Credit Risk Prediction")
st.write(
    "Enter the applicant's details to predict "
    "the likelihood of loan default."
)

st.divider()

# Input form
with st.form("credit_risk_form"):

    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=27
        )

        income = st.number_input(
            "Annual Income",
            min_value=0,
            value=65000
        )

        employment_length = st.number_input(
            "Employment Length (Years)",
            min_value=0.0,
            value=4.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=10000
        )

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            value=10.99
        )

        loan_percent_income = st.number_input(
            "Loan-to-Income Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.17,
            step=0.01
        )

    with col2:
        credit_history = st.number_input(
            "Credit History Length (Years)",
            min_value=0.0,
            value=5.0
        )

        home_ownership = st.selectbox(
            "Home Ownership",
            one.categories_[0]
        )

        loan_intent = st.selectbox(
            "Loan Purpose",
            one.categories_[1]
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            list(grade_mapping.keys())
        )

        previous_default = st.selectbox(
            "Previous Default on File",
            one.categories_[2]
        )

    submitted = st.form_submit_button(
        "Predict Credit Risk"
    )

# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home_ownership,
        "person_emp_length": employment_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amount,
        "loan_int_rate": interest_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": previous_default,
        "cb_person_cred_hist_length": credit_history
    }])

    # Handle missing values
    for col, median in missing_values.items():
        input_data[col] = input_data[col].fillna(median)

    # Encode loan grade
    input_data["loan_grade"] = (
        input_data["loan_grade"].map(grade_mapping)
    )

    # One-hot encode categorical features
    ohe_cols = [
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file"
    ]

    encoded = one.transform(input_data[ohe_cols])

    encoded_cols = one.get_feature_names_out(ohe_cols)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=input_data.index
    )

    input_data = pd.concat(
        [input_data.drop(columns=ohe_cols), encoded_df],
        axis=1
    )

    # Scale numerical features
    input_data[num_cols] = ss.transform(input_data[num_cols])

    # Match exact training feature order
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict using the combined Voting Classifier
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Estimated Default Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ Prediction: Loan Default")
    else:
        st.success("✅ Prediction: No Loan Default")

    st.caption(
        "This is a machine-learning prediction, "
        "not a guarantee of the applicant's future repayment behaviour."
    )

