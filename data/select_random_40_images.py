import os
import random

def retain_random_40_images_and_count(base_folder):
    for class_folder in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_folder)
        if os.path.isdir(class_path):
            # Get all files in the class folder
            files = os.listdir(class_path)
            
            if len(files) > 40:
                # Randomly select 40 files to retain
                files_to_retain = random.sample(files, 40)
                # Determine files to delete
                files_to_delete = [file for file in files if file not in files_to_retain]
                
                for file in files_to_delete:
                    file_path = os.path.join(class_path, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
            
            # Count remaining files
            remaining_files = os.listdir(class_path)
            total_images = len(remaining_files)
            print(f"Total images in '{class_folder}': {total_images}")

base_folder = "Dataset/"
retain_random_40_images_and_count(base_folder)
