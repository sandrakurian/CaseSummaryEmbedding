import os
from dotenv import load_dotenv
import requests
import json

from sentence_transformers import SentenceTransformer
from common_fx import process_file_content, setup_logger

# Set up the logger by calling the setup_logger function
logger = setup_logger()

# Load .env file
load_dotenv()
# get API key from https://aistudio.google.com/app/apikey?_gl=1*fx1u9i*_ga*MTY0MDI0ODg4LjE3MzY4NjM5NTU_ga_P1DBVKWT6V*MTczNjg2Mzk1NC4xLjAuMTczNjg2Mzk1NC42MC4wLjkwMDI1MzA1Mg..
api_key = os.getenv("SECRET_KEY")
if api_key:
    logger.info("Secret key loaded successfully.")
else:
    logger.error("Secret key not found. Did you forget to set up the .env file?")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"


# Initialize the model
model = SentenceTransformer('all-mpnet-base-v2')

def main(main_case, similar_cases):

    similar_cases_str = ""
    for case in similar_cases:
        similar_cases_str += f"{case}: {process_file_content(case, short=False)}\n"


    prompt = (f"Analyze the following main case summary and three similar case summaries. Based on the provided information, predict possible future outcomes with a detailed analysis, and justify these predictions with logical reasoning. Please structure the response as follows:\n\n1. **Analysis**: In this section, provide a comprehensive analysis of the main case and its possible future outcomes, supported by reasoning.\n2. **Predictions**: List your predictions in bullet points, making sure to explicitly state the possible future outcomes based on the case information.\n3. **Referrals**: In this section, explain how the similar cases were used to enhance the prediction and analysis for the main case. Discuss the similarities and differences that influenced the predictions.\n\n**Main Case Summary:**\n{process_file_content(main_case, short=False)}\n\n**Similar Case Summaries:**\n{similar_cases_str}")

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_json = response.json()
        # Extract the content text from the response
        text = response_json['candidates'][0]['content']['parts'][0]['text']
        # Replace \n with actual newlines
        formatted_text = text.replace("\\n", "\n")
        logger.info(formatted_text)
    else:
        logger.error(f"Error (predict.py): {response.status_code}, {response.text}")