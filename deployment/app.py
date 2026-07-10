
"""
Tourism Package Prediction
Streamlit Application
"""

import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# ==========================================================
# Download Registered Model
# ==========================================================

MODEL_REPO = "RajnishMohan/tourism-package-prediction-model"

MODEL_FILE = "tourism_package_prediction_model.joblib"

model_path = hf_hub_download(

    repo_id=MODEL_REPO,

    filename=MODEL_FILE

)

model = joblib.load(model_path)

# ==========================================================
# Streamlit Configuration
# ==========================================================

st.set_page_config(

    page_title="Tourism Package Prediction",

    page_icon="🌍",

    layout="wide"

)

st.title("🌍 Tourism Package Prediction")

st.markdown(

"""
Predict whether a customer is likely to purchase
the Wellness Tourism Package.
"""

)

# ==========================================================
# Customer Information
# ==========================================================

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    Occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    Designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        value=50000
    )

with col2:

    CityTier = st.selectbox(
        "City Tier",
        [1,2,3]
    )

    Passport = st.selectbox(
        "Passport",
        [0,1]
    )

    OwnCar = st.selectbox(
        "Own Car",
        [0,1]
    )

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [3,4,5]
    )

# ==========================================================
# Travel Information
# ==========================================================

st.header("Travel Information")

col3, col4 = st.columns(2)

with col3:

    NumberOfTrips = st.number_input(
        "Number of Trips",
        value=2
    )

    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        value=2
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        value=0
    )

with col4:

    ProductPitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        [
            "Company Invited",
            "Self Inquiry"
        ]
    )

# ==========================================================
# Sales Information
# ==========================================================

st.header("Sales Information")

col5, col6 = st.columns(2)

with col5:

    NumberOfFollowups = st.number_input(
        "Number of Followups",
        value=2
    )

    DurationOfPitch = st.number_input(
        "Duration Of Pitch",
        value=15
    )

with col6:

    PitchSatisfactionScore = st.slider(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

# ==========================================================
# Prediction
# ==========================================================

if st.button("Predict"):

    input_df = pd.DataFrame({

        "Age":[Age],

        "TypeofContact":[TypeofContact],

        "CityTier":[CityTier],

        "Occupation":[Occupation],

        "Gender":[Gender],

        "NumberOfPersonVisiting":[NumberOfPersonVisiting],

        "PreferredPropertyStar":[PreferredPropertyStar],

        "MaritalStatus":[MaritalStatus],

        "NumberOfTrips":[NumberOfTrips],

        "Passport":[Passport],

        "OwnCar":[OwnCar],

        "NumberOfChildrenVisiting":[NumberOfChildrenVisiting],

        "Designation":[Designation],

        "MonthlyIncome":[MonthlyIncome],

        "PitchSatisfactionScore":[PitchSatisfactionScore],

        "ProductPitched":[ProductPitched],

        "NumberOfFollowups":[NumberOfFollowups],

        "DurationOfPitch":[DurationOfPitch]

    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:

        st.success(
            "Customer is likely to purchase the package."
        )

    else:

        st.error(
            "Customer is unlikely to purchase the package."
        )

    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )

    st.subheader("Input Data")

    st.dataframe(input_df)

