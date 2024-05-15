import os

def concatenate_files(directory_path, output_file):
    with open(output_file, 'w') as output:
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    output.write(file.read())
                    output.write('\n')  # Add a newline between files

if __name__ == "__main__":
    # Provide the directory path containing files
    directory_path = input("Enter the directory path: ")

    # Provide the name of the output file
    output_file = input("Enter the name of the output file: ")

    concatenate_files(directory_path, output_file)
    print("Files concatenated successfully!")
