from kon_sequencer.paths import *
import os
import shutil

# Define the path to the main folder and the target folder
main_folder = TEST_PRIORSET_DIR
target_folder = TEST_PRIORSET_WAVONLY_DIR

# Ensure the target folder exists, create it if it doesn't
os.makedirs(target_folder, exist_ok=True)

#iterate through folders, for each folder copy "sum_" file into a dest folder

# Iterate through each subfolder in the main folder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)

    # Check if it's a directory and matches the pattern 'loop_X'
    if os.path.isdir(subfolder_path) and subfolder.startswith('loop_'):
        # Look for files in the subfolder
        for file in os.listdir(subfolder_path):
            if file.startswith('sum_track_'):
                file_path = os.path.join(subfolder_path, file)
                
                # Create a new file name that includes the subfolder name
                new_filename = f"{subfolder}_{file}"
                new_file_path = os.path.join(target_folder, new_filename)

                # Copy the file to the target folder with the new name
                shutil.copy(file_path, new_file_path)
                print(f"Copied and renamed: {file_path} -> {new_file_path}")

print("All sum track files have been copied.")