
import joblib
import pandas as pd
import streamlit as st

from huggingface_hub import hf_hub_download

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("Tourism Package Prediction")
st.write(
    "Predict whether a customer is likely to purchase the Wellness Tourism Package."
)

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

MODEL_REPO = "RajnishMohan/tourism-package-prediction-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="tourism_package_prediction_pipeline.pkl"
)

model = joblib.load(model_path)

# ------------------------------------------------------------
# User Inputs
# ------------------------------------------------------------

age = st.number_input("Age", 18, 100, 35)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

citytier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

numberofpersonvisiting = st.number_input(
    "Number of Persons Visiting",
    1,
    10,
    2
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [3,4,5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single","Married","Divorced"]
)

numberoftrips = st.number_input(
    "Number of Trips",
    0,
    30,
    2
)

passport = st.selectbox(
    "Passport",
    [0,1]
)

owncar = st.selectbox(
    "Own Car",
    [0,1]
)

numberofchildrenvisiting = st.number_input(
    "Children Visiting",
    0,
    5,
    0
)

designation = st.selectbox(
    "Designation",
    [
        "AVP",
        "VP",
        "Manager",
        "Senior Manager",
        "Executive"
    ]
)

monthlyincome = st.number_input(
    "Monthly Income",
    1000,
    500000,
    30000
)

pitchsatisfactionscore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

productpitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

numberoffollowups = st.number_input(
    "Number of Follow Ups",
    0,
    10,
    2
)

durationofpitch = st.number_input(
    "Duration of Pitch",
    5,
    120,
    20
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Age":[age],
        "TypeofContact":[typeofcontact],
        "CityTier":[citytier],
        "Occupation":[occupation],
        "Gender":[gender],
        "NumberOfPersonVisiting":[numberofpersonvisiting],
        "PreferredPropertyStar":[preferredpropertystar],
        "MaritalStatus":[maritalstatus],
        "NumberOfTrips":[numberoftrips],
        "Passport":[passport],
        "OwnCar":[owncar],
        "NumberOfChildrenVisiting":[numberofchildrenvisiting],
        "Designation":[designation],
        "MonthlyIncome":[monthlyincome],
        "PitchSatisfactionScore":[pitchsatisfactionscore],
        "ProductPitched":[productpitched],
        "NumberOfFollowups":[numberoffollowups],
        "DurationOfPitch":[durationofpitch]
    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success(
            f"Customer is likely to purchase the package.\n\nProbability: {probability:.2%}"
        )
    else:
        st.error(
            f"Customer is unlikely to purchase the package.\n\nProbability: {probability:.2%}"
        )

