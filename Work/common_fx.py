import os

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
        print(f"Error: The file '{file_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
