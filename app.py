import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load your trained model (ensure you have a trained model saved as 'model.pkl')
# Load the model from the file
model =load("movie_rating_prediction.joblib")

# Use the loaded model to make predictions


# Title of the web app
st.title('Movie Success Predictor')

# Input features
year = st.number_input('Year', min_value=1900, max_value=2100, step=1)
actor_1 = st.text_input('Actor 1')
actor_2 = st.text_input('Actor 2')
duration = st.number_input('Duration (minutes)', min_value=1, step=1)
genre = st.text_input('Genre')
votes = st.number_input('Votes', min_value=0, step=1)
director = st.text_input('Director')

# Make predictionc
if st.button('Predict'):
    # Create feature array
    features = pd.DataFrame({
    'Year': [year],
    'Actor 1': [actor_1],
    'Actor 2': [actor_2],
    'Duration': [duration],
    'Genre': [genre],
    'Votes': [votes],
    'Director': [director]
})
    # features = np.array([year, actor_1, actor_2, duration, genre, votes, director])
    
    # Assuming you have some preprocessing steps, apply them here before prediction
    # For example, converting categorical variables to numerical, scaling, etc.
    # processed_features = preprocess_features(features)
    
    # For this example, let's assume the features are correctly formatted for the model
    prediction = model.predict(features)
    
    st.write(f'Predicted Success: {prediction[0]}')

# To run the Streamlit app, use the command: streamlit run app.py
