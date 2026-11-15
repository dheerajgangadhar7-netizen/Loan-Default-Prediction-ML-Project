
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# LOAD SAVED ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = joblib.load("credit_risk_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    shap_background = joblib.load("shap_background.pkl")

    return model, encoder, scaler, shap_background


model, one, ss, shap_background = load_artifacts()

# Exact feature order used during model training
feature_columns = list(model.estimators_[0].feature_names_in_)

# Numerical features scaled during training
num_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

# Categorical features one-hot encoded during training
ohe_cols = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]

# Loan grade LabelEncoder mapping
grade_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

# Ensure SHAP background has the same feature order
shap_background = shap_background.reindex(
    columns=feature_columns,
    fill_value=0
)

# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.title("🏦 Loan Default Prediction")

st.write(
    "Enter the applicant's details to predict "
    "the likelihood of loan default."
)

st.divider()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

with st.form("loan_prediction_form"):

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
        "Predict Loan Default"
    )

# --------------------------------------------------
# PREPROCESSING AND PREDICTION
# --------------------------------------------------

if submitted:

    # Create input DataFrame
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

    # Encode loan grade
    input_data["loan_grade"] = (
        input_data["loan_grade"].map(grade_mapping)
    )

    # One-hot encode categorical features
    encoded = one.transform(input_data[ohe_cols])

    encoded_cols = one.get_feature_names_out(ohe_cols)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=input_data.index
    )

    input_data = pd.concat(
        [
            input_data.drop(columns=ohe_cols),
            encoded_df
        ],
        axis=1
    )

    # Scale numerical features
    input_data[num_cols] = ss.transform(
        input_data[num_cols]
    )

    # Match exact training feature order
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

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

    # --------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------

    st.divider()

    st.subheader("Why did the model make this prediction?")

    st.write(
        "The chart below shows which features contributed "
        "to the predicted loan default probability."
    )

    with st.spinner("Calculating SHAP explanation..."):

        # Prediction function for SHAP
        def predict_probability(data):

            data = pd.DataFrame(
                data,
                columns=feature_columns
            )

            return model.predict_proba(data)[:, 1]

        # Create KernelExplainer
        explainer = shap.KernelExplainer(
            predict_probability,
            shap_background
        )

        # Calculate SHAP values for this applicant
        shap_values = explainer.shap_values(
            input_data,
            nsamples=100
        )

        # Extract class 1 (loan default) SHAP values
        if isinstance(shap_values, list):

            default_shap_values = np.asarray(
                shap_values[1]
            )[0]

        else:

            shap_array = np.asarray(shap_values)

            if shap_array.ndim == 3:

                if shap_array.shape[-1] == 2:
                    shap_array = shap_array[:, :, 1]

                elif shap_array.shape[0] == 2:
                    shap_array = shap_array[1]

            default_shap_values = shap_array[0]

        # Create SHAP importance DataFrame
        shap_df = pd.DataFrame({
            "Feature": feature_columns,
            "SHAP Value": default_shap_values
        })

        # Select 10 most influential features
        shap_df["Absolute Value"] = (
            shap_df["SHAP Value"].abs()
        )

        shap_df = shap_df.sort_values(
            by="Absolute Value",
            ascending=True
        ).tail(10)

        # Plot SHAP results
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            shap_df["Feature"],
            shap_df["SHAP Value"]
        )

        ax.axvline(
            x=0,
            linewidth=1
        )

        ax.set_xlabel(
            "SHAP Value (effect on default probability)"
        )

        ax.set_title(
            "Top Features Influencing Loan Default Prediction"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    st.info(
        "Positive SHAP values push the prediction towards a higher "
        "loan default probability. Negative SHAP values push it "
        "towards a lower loan default probability."
    )
