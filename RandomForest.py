import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    with open("random_forest_regressor.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Random Forest Predictor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Random Forest Regressor")
st.write("Enter the details below to get a prediction.")

# -----------------------------
# Input fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=25,
        step=1
    )

    experience = st.number_input(
        "Experience (Years)",
        min_value=0,
        max_value=60,
        value=3,
        step=1
    )

    education = st.number_input(
        "Education (Years)",
        min_value=0,
        max_value=30,
        value=16,
        step=1
    )

    hours = st.number_input(
        "Hours Per Week",
        min_value=0,
        max_value=168,
        value=40,
        step=1
    )

with col2:
    projects = st.number_input(
        "Projects Completed",
        min_value=0,
        max_value=1000,
        value=5,
        step=1
    )

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )

    performance = st.number_input(
        "Performance Score",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=0.1
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict", use_container_width=True):

    input_data = pd.DataFrame({
        "Age": [age],
        "Experience_Years": [experience],
        "Education_Years": [education],
        "Hours_Per_Week": [hours],
        "Projects_Completed": [projects],
        "Certifications": [certifications],
        "Performance_Score": [performance]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Predicted Value",
        value=f"{prediction:.2f}"
    )

    # Show input data
    with st.expander("View Input Data"):
        st.dataframe(input_data)