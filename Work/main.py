import os
from embed import main as embed_main  # Import the main function from embed.py
from common_fx import process_file_content as process_file_content


def read_text_files(base_path, target_folder):
    """
    Reads all text files from a specified folder and returns their names and content.

    :param base_path: Base directory from which the target folder is located.
    :param target_folder: Relative path to the folder containing text files.
    :return: Dictionary where keys are file names and values are file contents.
    """
    folder_path = os.path.join(base_path, target_folder)
    file_data = {}

    if not os.path.exists(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist.")
        return file_data

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        file_data[file_name] = process_file_content (file_name, short = False)

    return file_data

def summary(target_folder):
    # Read files
    files_content = read_text_files(base_path, target_folder)

    # Display file names and content
    for file_name, content in files_content.items():
        # print(f"File: {file_name}\nContent:\n{content[:50]}")
        embed_main(file_name, content, "summary")

if __name__ == "__main__":
    # Base directory of the script
    base_path = os.path.abspath(os.path.dirname(__file__))

    # Relative path to the folder containing text files
    target_folder = "C:\\Users\Kurian-Sandra\\Desktop\\CaseSummaryEmbedding\\ExampleCases"

    summary(target_folder)

    print("-----")

    top_similar_cases = embed_main("Sarah.txt", None, "sandra")
    for item in top_similar_cases:
        print(item[0])  # Accessing the first element in each tuple


    print("-----")
