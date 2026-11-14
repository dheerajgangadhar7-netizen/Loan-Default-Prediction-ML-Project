
import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing objects
@st.cache_resource
def load_artifact():
    return joblib.load("loan_default_app.pkl")

artifact = load_artifact()

model = artifact["model"]
lab = artifact["label_encoder"]
one = artifact["onehot_encoder"]
ss = artifact["scaler"]

num_cols = artifact["num_cols"]
ohe_cols = artifact["ohe_cols"]
drop_cols = artifact["drop_cols"]
feature_columns = artifact["feature_columns"]


# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Default Prediction")
st.write("Enter the applicant's details to predict the likelihood of loan default.")

st.divider()

# Input form
with st.form("loan_form"):

    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income", min_value=0, value=50000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        months_employed = st.number_input("Months Employed", min_value=0, value=24)
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)
        interest_rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1, value=36)
        dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=0.3)

    with col2:
        education = st.selectbox("Education", lab.classes_)

        employment = st.selectbox(
            "Employment Type", one.categories_[0]
        )
        marital = st.selectbox(
            "Marital Status", one.categories_[1]
        )
        mortgage = st.selectbox(
            "Has Mortgage", one.categories_[2]
        )
        dependents = st.selectbox(
            "Has Dependents", one.categories_[3]
        )
        purpose = st.selectbox(
            "Loan Purpose", one.categories_[4]
        )
        cosigner = st.selectbox(
            "Has Co-Signer", one.categories_[5]
        )

    submitted = st.form_submit_button("Predict Loan Default")


# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment,
        "MaritalStatus": marital,
        "HasMortgage": mortgage,
        "HasDependents": dependents,
        "LoanPurpose": purpose,
        "HasCoSigner": cosigner
    }])

    # Label encode Education
    input_data["Education"] = lab.transform(input_data["Education"])

    # One-hot encode categorical features
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

    # Drop reference categories
    input_data = input_data.drop(columns=drop_cols)

    # Scale numerical features
    input_data[num_cols] = ss.transform(input_data[num_cols])

    # Match exact training feature order
    input_data = input_data.reindex(columns=feature_columns)

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    st.metric("Estimated Default Probability", f"{probability * 100:.2f}%")

    if prediction == 1:
        st.error("⚠️ Prediction: Loan Default")
    else:
        st.success("✅ Prediction: No Loan Default")

    st.caption(
        "This is a machine-learning prediction, not a guarantee of the applicant's future repayment behaviour."
    )
