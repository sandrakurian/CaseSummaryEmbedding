import os
import logging
from datetime import datetime

log_file = "script.log"  # Define your log file path here

# Function to log the header with current time
def log_header(logger):
    current_time = datetime.now().strftime("%H:%M")  # Get current time in hh:mm format
    logger.info(f"\n========== Log Session Started at {current_time} ==========\n")

# Set up logging configuration
def setup_logger():
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(message)s"  # Only log the message without timestamp and level by default
    )
    logger = logging.getLogger()
    
    # Log the header only once at the start of the log session
    log_header(logger)
    
    return logger


#from common_fx import read_text_files as read_text_files
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
        print(f"Error (common_fx.py): The directory '{folder_path}' does not exist.")
        return file_data

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        file_data[file_name] = process_file_content (file_name, short = False)

    return file_data

#from common_fx import process_file_content as process_file_content
def process_file_content(file_name, short = True, file_path = "C:\\Users\Kurian-Sandra\\Desktop\\CaseSummaryEmbedding\\ExampleCases"):
    try:
        with open(file_path+"\\"+file_name, 'r') as file:
            if short:
                # Read the file content
                content = file.read()
                # Extract content between "### Case Summary" and "### Health-Related Issues"
                start_marker = "### Case Summary"
                end_marker = "### Health-Related Issues"
                
                start_index = content.find(start_marker)
                end_index = content.find(end_marker)

                if start_index != -1 and end_index != -1:
                    # Extract the required content
                    content = content[start_index + len(start_marker):end_index].strip()
                else:
                    print("Markers not found in the file.")
            else:
                # Read all file content
                content = file.read()
                # Split content into paragraphs
                paragraphs = content.split('\n\n')
                # Remove the first paragraph
                content = '\n\n'.join(paragraphs[1:])
            return content
    except FileNotFoundError:
        print(f"Error (common_fx.py): The file '{file_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred (common_fx.py): {e}")
