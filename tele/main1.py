import pandas as pd

# Load the dataframe from a file
df = pd.read_csv('Churn.csv')

# Print the column names in the dataframe
print('Column names in dataframe:', df.columns)

# Create the Streamlit app
import streamlit as st

# Print the column names in the Streamlit app
print('Column names in Streamlit app:', st.session_state.column_names)