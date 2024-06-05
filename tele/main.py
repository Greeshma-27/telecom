# popularity_predictor_app.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('Churn.csv')


# Features and target variable
numeric_features = ['Al','Total_day_minutes','Total_day_calls','Total_day_charge','Total_night_minutes','Total_night_calls','Total_night_charge','Customer_service_calls']
target = 'Churn'

# Ensure only numeric features are included
df_numeric = df[numeric_features + [target]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_numeric.drop(columns=[target]), df_numeric[target], test_size=0.2, random_state=42)

# Train a Random Forest Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Function to predict popularity
def predict_popularity(Al,Total_day_minutes,Total_day_calls,Total_day_charge,Total_night_minutes,Total_night_calls,Total_night_charge,Customer_service_calls):
    input_data = pd.DataFrame({
        'Al':[Al],
        'Total_day_minutes': [Total_day_minutes],
        'Total_day_calls': [Total_day_calls],
        'Total_day_charge': [Total_day_charge],
        'Total_night_minutes': [Total_night_minutes],
        'Total_night_calls': [Total_night_calls],
        'Total_night_charge': [Total_night_charge],
        'Customer_service_calls': [Customer_service_calls]
    })

    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title('Churn Predictor')

# Image



# Input form for user to enter feature values
Al=st.slider('Al',min_value=1,max_value=500,step=1)
Total_day_minutes = st.slider('Total_day_minutes', min_value=0, max_value=350, step=1)
Total_day_calls = st.slider('Total_day_calls', min_value=0, max_value=160, step=1)
Total_day_charge = st.slider('Total_day_charge', min_value=0, max_value=60, step=1)
Total_night_minutes = st.slider('Total_night_minutes', min_value=0, max_value=395, step=1)
Total_night_calls = st.slider('Total_night_calls', min_value=0, max_value=166, step=1)
Total_night_charge = st.slider('Total_night_charge', min_value=0, max_value=200, step=1)
Customer_service_calls = st.number_input('Customer_service_calls', min_value=0, max_value=9, step=1)
# Predict button
if st.button('Predict Churn'):
    prediction = predict_popularity(Al,Total_day_minutes,Total_day_calls,Total_day_charge,Total_night_minutes,Total_night_calls,Total_night_charge,Customer_service_calls)
    if prediction:
        st.success('Predicted result: The employee is ready to churn.')
    else:
        st.success('Predicted result: The employee is not ready to churn.')

