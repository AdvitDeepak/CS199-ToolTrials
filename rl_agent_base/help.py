def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    length = len(content)
    # Remove newlines and whitespace characters
    content = [line.strip() for line in content]
    return content, length 

