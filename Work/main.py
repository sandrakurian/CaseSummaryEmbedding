import os
from embed import main as embed_main  # Import the main function from embed.py
# from your_module import read_text_files  # Replace `your_module` with the module containing read_text_files

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

        # Only process .txt files
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Split the content into paragraphs based on double newlines
                    paragraphs = content.split('\n\n')
                    # Remove the first two paragraphs and join the rest back together
                    content = '\n\n'.join(paragraphs[1:])
                    file_data[file_name] = content
            except Exception as e:
                print(f"Error reading file '{file_name}': {e}")
    return file_data

if __name__ == "__main__":
    # Base directory of the script
    base_path = os.path.abspath(os.path.dirname(__file__))

    # Relative path to the folder containing text files
    target_folder = "C:\\Users\Kurian-Sandra\\Desktop\\CaseSummaryEmbedding\\ExampleCases"

    # Read files
    files_content = read_text_files(base_path, target_folder)

    # Display file names and content
    for file_name, content in files_content.items():
        # print(f"File: {file_name}\nContent:\n{content[:50]}")
        embed_main(file_name, content, "summary")

    print("-----")
