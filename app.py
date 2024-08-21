import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Load environment variables (if any)
load_dotenv()

# Get OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Create an LLM instance
llm = OpenAI(api_token=openai_api_key)

# Create PandasAI instance
pandas_ai = PandasAI(llm)

# Streamlit app title
st.title("Prompt-driven data analysis with PandasAI")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.write("Data Preview:")
    st.write(df.head())

    # Prompt input area
    prompt = st.text_area("Enter your prompt:")

    # Generate button
    if st.button("Generate"):
        if prompt:
            # Generate response using PandasAI
            with st.spinner("Generating response..."):
                try:
                    response = pandas_ai.run(df, prompt)
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a prompt.")
