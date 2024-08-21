import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()

# Get OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit app title
st.title("Chat with Dataset 💳")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset directly from the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Show dataframe preview
    with st.expander("🔍 Dataframe Preview"):
        st.write(df.tail(3))

    # Input query
    query = st.text_area("💬 Chat with Dataframe")

    # Submit button
    if st.button("Submit"):
        if query:
            # Initialize the LLM with your OpenAI API key
            llm = OpenAI(api_key=openai_api_key)
            query_engine = SmartDataframe(df, config={"llm": llm})
            
            # Get the answer from the query engine
            answer = query_engine.chat(query)
            
            # Display the answer
            st.write(answer)
        else:
            st.warning("Please enter a query.")
else:
    st.write("Please upload a CSV file to start.")
