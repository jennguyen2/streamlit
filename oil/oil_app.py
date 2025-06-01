import streamlit as st
import pandas as pd
import numpy as np

# # Streamlit app title
st.title("Pandas Profiling App")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Generate pandas profiling report
    st.write("Generating Pandas Profiling Report...")
    # profile = ProfileReport(df, explorative=True)
    summary = df.describe(include='all')
    st.write("Summary Statistics:")
    st.write(summary)
    
    # Display the report in Streamlit
    # st_profile_report(profile)
else:
    st.write("Please upload a CSV file to get started.")
    

