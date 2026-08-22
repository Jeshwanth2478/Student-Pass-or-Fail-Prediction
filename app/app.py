import streamlit as st
import requests


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# Application Title
# ==========================================

st.title(
    "🎓 Student Performance Predictor"
)

st.write(
    "Enter student information to predict "
    "whether the student is likely to pass."
)


# ==========================================
# Input Fields
# ==========================================

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)


attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)


previous_score = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)


assignments_completed = st.number_input(
    "Assignments Completed",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)


# ==========================================
# Prediction Button
# ==========================================

if st.button(
    "Predict Performance"
):

    # --------------------------------------
    # Prepare API Request
    # --------------------------------------

    data = {
        "study_hours": study_hours,

        "attendance": attendance,

        "previous_score": previous_score,

        "assignment_completed":
            assignments_completed
    }

    try:

        # ----------------------------------
        # Send request to FastAPI
        # ----------------------------------

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        # ----------------------------------
        # Process Response
        # ----------------------------------

        if response.status_code == 200:

            result = response.json()

            prediction = result[
                "prediction"
            ]

            probability = result[
                "pass_probability"
            ]

            # ------------------------------
            # Display Result
            # ------------------------------

            if prediction == "Pass":

                st.success(
                    f"Prediction: {prediction}"
                )

            else:

                st.error(
                    f"Prediction: {prediction}"
                )

            st.info(
                f"Pass Probability: "
                f"{probability:.2%}"
            )

        else:

            st.error(
                f"API Error: "
                f"{response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI. "
            "Make sure the API server is running."
        )