import os

from common_fx import process_file_content, setup_logger, generate_content

# Set up the logger by calling the setup_logger function
logger = setup_logger()

def main(main_case, similar_cases):

    similar_cases_str = ""
    for case,_ in similar_cases:
        similar_cases_str += f"\n\nCASE: {case}\n{process_file_content(case, False)}"

    prompt = (f"Analyze the following main case summary and three similar case summaries. Based on the provided information, predict possible future outcomes with a detailed analysis, and justify these predictions with logical reasoning. Please structure the response as follows:\n\n\n1. **Analysis**: In this section, provide a comprehensive analysis of the main case and its possible future outcomes, supported by reasoning.\n\n2. **Predictions**: List your predictions in bullet points, making sure to explicitly state the possible future outcomes based on the case information.\n\n3. **Referrals**: In this section, explain how the similar cases were used to enhance the prediction and analysis for the main case. Discuss the similarities and differences that influenced the predictions.\n\n\n**Main Case Summary:**\n\n{process_file_content(main_case, short=False)}\n\n\n**Similar Case Summaries:**\n\n{similar_cases_str}")

    output = generate_content(prompt)
    return output