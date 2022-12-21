import streamlit as st
import pandas as pd
import joblib

st.title("Streamlit ML App for Pets!")

# Add a text input
height = st.number_input("Enter Height", min_value=0.00)
weight = st.number_input("Enter Weight", min_value=0.00)
eye_colour = st.selectbox("Select Eye Colour", ('Blue', 'Brown'))

# If button is pressed
if st.button('Submit'):

    # Unpickle classifier
    pet_model = joblib.load("pet_model.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[height, weight, eye_colour]],
                    columns = ["Height", "Weight", "Eye"])
    X = X.replace(["Brown", "Blue"], [1,0])

    # Get prediction
    prediction = pet_model.predict(X)[0]

    st.text(f"This instance is a {prediction}")