# -*- coding: utf-8 -*-
"""part1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DSB_OknLFJeHKGmr3wC_HFE7ICV_x919
"""
import streamlit as st
import json
import os
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go

def download_10k_filings(ticker, start_year):
    """
    Downloads and extracts 10-K filings for the specified ticker and starting year until the current year.
    This function will clone the EDGAR crawler repository, download the requirements.txt if it's not present,
    set up configurations, and run the download and extract scripts.

    Args:
        ticker (str): The stock ticker symbol for the company (e.g., "AAPL").
        start_year (int): The year from which to start downloading filings.
    """
    # Get the current year
    from datetime import datetime
    current_year = datetime.now().year

    # Clone the repository
    repo_url = "https://github.com/nlpaueb/edgar-crawler.git"
    repo_dir = "edgar-crawler"

    # Clone the repository only if it doesn't exist
    if not os.path.exists(repo_dir):
        print(f"Cloning the repository from {repo_url}...")
        subprocess.run(["git", "clone", repo_url], check=True)

    # Navigate to the repository directory
    os.chdir(repo_dir)

    # Download the requirements.txt only if it doesn't exist
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"Downloading {requirements_file}...")
        subprocess.run(["curl", "-O", "https://raw.githubusercontent.com/nlpaueb/edgar-crawler/master/requirements.txt"], check=True)

    # Install the required dependencies if not already installed
    print("Installing required dependencies...")
    subprocess.run(["pip", "install", "-r", requirements_file], check=True)

    # Create the configuration dictionary
    config = {
        "download_filings": {
            "start_year": start_year,
            "end_year": current_year,
            "quarters": [1, 2, 3, 4],
            "filing_types": ["10-K"],
            "cik_tickers": [ticker],  # Dynamic ticker
            "user_agent": "Your Name (your-email@example.com)",  # Update with your information
            "raw_filings_folder": "RAW_FILINGS",
            "indices_folder": "INDICES",
            "filings_metadata_file": "FILINGS_METADATA.csv",
            "skip_present_indices": True
        },
        "extract_items": {
            "raw_filings_folder": "RAW_FILINGS",
            "extracted_filings_folder": "EXTRACTED_FILINGS",
            "filings_metadata_file": "FILINGS_METADATA.csv",
            "filing_types": ["10-K"],
            "include_signature": False,
            "items_to_extract": [],
            "remove_tables": True,
            "skip_extracted_filings": True
        }
    }

    # Write the config to a file
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Run the download and extract scripts
    try:
        print(f"Downloading filings for {ticker} from {start_year} to {current_year}...")
        subprocess.run(["python", "download_filings.py"], check=True)

        print(f"Extracting items from filings for {ticker}...")
        subprocess.run(["python", "extract_items.py"], check=True)

        print("Process completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    finally:
        # Navigate back to the original directory
        os.chdir('..')

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Sentiment Analysis Function
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    if result['label'] == 'positive':
        return result['score']
    return 0

# Function to extract data from JSON files

def extract_all_json_content(folder_path):
    """
    Extracts all content from JSON files in the specified folder, using only the first three parts of the filename.

    Args:
        folder_path (str): Path to the folder containing the JSON files.

    Returns:
        list: A list of dictionaries containing the content of each JSON file.
    """
    extracted_content = []

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return extracted_content

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Process only JSON files
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Extract the first three components from the filename
                parts = file_name.replace(".json", "").split("_")[:3]
                if len(parts) < 3:
                    print(f"Skipping invalid filename: {file_name}")
                    continue

                cik, filing_type, year = parts

                # Load the JSON content
                with open(file_path, 'r') as f:
                    content = json.load(f)

                # Add metadata to the content
                content["cik"] = cik
                content["filing_type"] = filing_type
                content["year"] = year

                # Append the content to the list
                extracted_content.append(content)
                print(f"Successfully extracted data from {file_name}")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    return extracted_content


# Main Streamlit App
st.title("EDGAR 10-K Filings Sentiment Analysis")

ticker = st.text_input("Enter Ticker Symbol:")
year = st.number_input("Enter Start Year:", min_value=2000, max_value=2025, step=1)

if st.button("Analyze"):
    if ticker and year:
        folder_path = f"edgar-crawler/datasets/EXTRACTED_FILINGS/10-K"
        data = extract_all_json_content(folder_path)
        
        company_dfs = {}
        for report in data:
            company_name = report.get('company', 'Unknown')
            report_year = report.get('year', 'Unknown')
            if company_name not in company_dfs:
                company_dfs[company_name] = pd.DataFrame(columns=['year'] + [f'item_{i}' for i in range(1, 17)])
            row = {'year': report_year}
            for item in range(1, 17):
                item_key = f'item_{item}'
                if item_key in report:
                    row[item_key] = analyze_sentiment(report[item_key])
                else:
                    row[item_key] = None
            company_dfs[company_name] = pd.concat([company_dfs[company_name], pd.DataFrame([row])], ignore_index=True)
        
        for company, df in company_dfs.items():
            st.subheader(f"Sentiment Scores for {company}")
            st.dataframe(df)
            df['Average'] = df.mean(axis=1)
            fig = px.line(df, x='year', y='Average', title=f"Sentiment Over Time for {company}")
            st.plotly_chart(fig)

            # Show descriptive statistics
            st.subheader("Descriptive Statistics")
            st.write(df.describe())

            # Correlation Matrix
            st.subheader("Correlation Matrix")
            corr_matrix = df.corr()
            fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='Viridis'))
            st.plotly_chart(fig_corr)
    else:
        st.error("Please enter both Ticker Symbol and Start Year.")

