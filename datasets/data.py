import os

# Set the path to the main dataset directory (adjust as needed)
dataset_dir = os.path.dirname(os.path.abspath(__file__))

# List all subfolders (disease folders)
disease_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
sum = 0
# Count files in each disease folder and print
for folder in disease_folders:
    folder_path = os.path.join(dataset_dir, folder)
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    sum= file_count + sum
    print(f"{folder}: {file_count}")
    print(f"Total number of files: {sum}")
