
import streamlit as st
import pickle
import pandas as pd
import sklearn



# load trained model 
with open("loanmodel.pkl", "rb") as file:
    model = pickle.load(file)


# App title
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Loan Approval</b></h1>",
    unsafe_allow_html=True
)

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numeric values #order way they were entered in the model
Requested_Loan_Amount = st.slider('Requested Loan Amount', min_value=1000, max_value=500000, step=1000.0)
FICO_score = st.slider('FICO Score', min_value=300, max_value=850, step=1.0)
Monthly_Gross_Income = st.slider('Monthly Gross Income', min_value=500, max_value=20000, step=100.0)
Monthly_Housing_Payment = st.slider('Monthly Housing Payment', min_value=500, max_value=20000, step=100.0)
Ever_Bankrupt_or_Foreclose = st.selectbox('Ever Bankrupt or Foreclosed', ['Yes', 'No'])

# Categorical Inputs with options
Reason = st.selectbox('Reason', ['Debt Consolidation', 'Home Improvement', 'Unexpected Cost', 'Major Purchase',
                                 'Credit Card Refinancing', 'Other'])
Employment_Status = st.selectbox('Employment Status', ['Full Time', 'Part Time', 'Unemployed'])



# --- Prepare Data for Prediction ---

input_data = pd.DataFrame([{
    'Requested_Loan_Amount': Requested_Loan_Amount,
    'FICO_score': FICO_score,
    'Monthly_Gross_Income': Monthly_Gross_Income,
    'Monthly_Housing_Payment': Monthly_Housing_Payment,
    'Ever_Bankrupt_or_Foreclose': Ever_Bankrupt_or_Foreclose,
    'Reason': Reason,
    'Employment_Status': Employment_Status}])

# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data, columns=['Reason', 'Employment_Status'])

# 2. Add any "missing" columns the model expects (fill with 0).
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Evaluate Loan"):
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]

    # Display result
    if prediction == 1:
        st.write("The prediction is: **Loan Approved** 💲")
    else:
        st.write("The prediction is: **Loan Denied** 🚫")
