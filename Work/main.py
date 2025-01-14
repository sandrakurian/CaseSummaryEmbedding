import os
from embed import main as embed_main  # Import the main function from embed.py
from predict import main as predict_main  # Import the main function from embed.py
from common_fx import process_file_content as process_file_content
from common_fx import read_text_files as read_text_files

def summary(target_folder):
    # Read files
    files_content = read_text_files(base_path, target_folder)

    # Display file names and content
    for file_name, content in files_content.items():
        # print(f"File: {file_name}\nContent:\n{content[:50]}")
        embed_main(file_name, content, "summary")

if __name__ == "__main__":
    # # Base directory of the script
    # base_path = os.path.abspath(os.path.dirname(__file__))

    # # Relative path to the folder containing text files
    # target_folder = "C:\\Users\Kurian-Sandra\\Desktop\\CaseSummaryEmbedding\\ExampleCases"

    # summary(target_folder)

    # print("-----")
    # case = "Sarah.txt"

    # top_similar_cases = embed_main(case, None, "sandra")
    # print("-----")

    # prediction = predict_main(case, top_similar_cases)
    prediction = predict_main("Sarah.txt", ["SarahJakeMia.txt", "SarahRachel.txt", "MaryBrown.txt"])
    print(prediction)

    print("-----")
