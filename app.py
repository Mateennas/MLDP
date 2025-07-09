import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model.pkl')

st.title('HDB Resale Price Prediction')\

# Define input options
towns = ['Tampines', 'Bedok', 'Punggol']
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM',]
storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09']

## User inputs
town_selected = st.selectbox("Select Town", towns)
flat_type_selected = st.selectbox("Select Flat Type", flat_types)
storey_range_selected = st.selectbox("Select Storey Range", storey_ranges)
floor_area_selected = st.slider("Select Floor Area (sqm)", min_value=30, max_value=200, value=70)

## predict button
if st.button("Predict HDB price"):

    # Create a Dict with the user inputs
    input_data = {
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area_sqm': floor_area_selected
    
    }

    #Convert input data to a Dataframe
    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area': [floor_area_selected]
    })
        
    # Convert categorical variables to dummy variables
    df_input = pd.get_dummies(df_input,
                               columns=['town', 'flat_type', 'storey_range'])

    # df_input = df_input.to_numpy()
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Prediccted Resale Price: ${y_unseen_pred:,.2f}")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://static.vecteezy.com/system/resources/thumbnails/026/185/327/small_2x/waterfall-and-stone-copy-space-blurred-background-ai-generated-photo.jpg"); 
        background-size: cover;
    }}
    </style>
    """
)