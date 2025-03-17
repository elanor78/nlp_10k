# -*- coding: utf-8 -*-
import streamlit as st
import json
import subprocess
import sys
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go


def download_10k_filings(ticker, start_year):
    from datetime import datetime
    current_year = datetime.now().year

    repo_url = "https://github.com/nlpaueb/edgar-crawler.git"
    repo_dir = "edgar-crawler"

    if not os.path.exists(repo_dir):
        st.write("Cloning the EDGAR crawler repository...")
        subprocess.run(["git", "clone", repo_url], check=True)

    requirements_file = os.path.join(repo_dir, "requirements.txt")
    if not os.path.exists(requirements_file):
        st.write("Downloading requirements.txt...")
        subprocess.run(["curl", "-o", requirements_file,
                        "https://raw.githubusercontent.com/nlpaueb/edgar-crawler/master/requirements.txt"], check=True)

    st.write("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)

    config = {
        "download_filings": {
            "start_year": start_year,
            "end_year": current_year,
            "quarters": [1, 2, 3, 4],
            "filing_types": ["10-K"],
            "cik_tickers": [ticker],
            "user_agent": "Your Name (your-email@example.com)",
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

    with open(os.path.join(repo_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    os.chdir(repo_dir)
    try:
        st.write("Downloading and extracting filings...")
        subprocess.run([sys.executable, "download_filings.py"], check=True)
        subprocess.run([sys.executable, "extract_items.py"], check=True)
        st.write("Process completed!")
    except subprocess.CalledProcessError as e:
        st.error(f"Error during process: {e}")
    finally:
        os.chdir('..')


model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def analyze_sentiment(text):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()[0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    return probabilities


def extract_all_json_content(folder_path):
    extracted_content = []
    if not os.path.exists(folder_path):
        st.error(f"The folder '{folder_path}' does not exist.")
        return extracted_content

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                cik, filing_type, year = file_name.replace(".json", "").split("_")[:3]
                content.update({"cik": cik, "filing_type": filing_type, "year": year})
                extracted_content.append(content)
            except (ValueError, json.JSONDecodeError) as e:
                st.warning(f"Error reading {file_name}: {e}")
    return extracted_content


st.title("EDGAR 10-K Filings Sentiment Analysis")
ticker = st.text_input("Enter Ticker Symbol:")
year = st.number_input("Enter Start Year:", min_value=2000, max_value=2025)

if st.button("Analyze") and ticker and year:
    folder_path = "edgar-crawler/EXTRACTED_FILINGS"
    data = extract_all_json_content(folder_path)
    if data:
        company_df = pd.DataFrame(data)
        st.write("Extracted Data:")
        st.dataframe(company_df)
    else:
        st.warning("No data found for the given inputs.")
