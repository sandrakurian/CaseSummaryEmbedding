import openai
openai.api_key = "sk-svcacct-QonuAKoxk2DIaQ3U6E_siY50OOCW5Z48MnX6SHd903jvupoLsvBxpvtZ3qeqcT3BlbkFJsX8Oh9UMZUEJSDRCGhCDcxZ2JMdmrq-pYKEHPBvXVeQXc4aPZ1ooMgu49VR1wA"

from common_fx import process_file_content as process_file_content

def main(main_case, similar_cases):
    # Prepare the input prompt for the AI
    prompt = (
        "Analyze the following case summary and three similar case summaries. "
        "Based on the information provided, predict future outcomes with a detailed analysis and justify the predictions with logical reasoning. "
        "Structure the response into separate paragraphs for analysis and predictions."
        "When referring to other cases put [1], [2], or [3] accordingly at the end of the sentence.\n\n"
        "Main Case Summary:\n"
        f"{process_file_content(main_case)}\n\n"
        "Similar Case Summaries:\n"
        f"1. {process_file_content(similar_cases[0])}\n"
        f"2. {process_file_content(similar_cases[1])}\n"
        f"3. {process_file_content(similar_cases[2])}\n"
    )

    # Call OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 instead of GPT-4
            messages=[{"role": "system", "content": "You are an AI analyst."},
                    {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )

        # Extract and return the AI's response
        return response.choices[0].message['content']

    except Exception as e:
        return f"Error in prediction: {str(e)}"
