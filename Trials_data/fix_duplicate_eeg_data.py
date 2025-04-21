import os
import pandas as pd
import glob
import sys
from pathlib import Path

def remove_duplicates_from_csv(file_path):
    """
    Read a CSV file, remove duplicated rows, and save it back.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Print original shape
        original_shape = df.shape
        print(f"Original shape: {original_shape}")
        
        # Find duplicate rows (all columns including counter and timestamp)
        df = df.drop_duplicates()
        
        # Print new shape
        new_shape = df.shape
        print(f"New shape: {new_shape}")
        
        # If duplicates were found and removed, save the file
        if original_shape[0] != new_shape[0]:
            # Create a backup of the original file
            backup_path = file_path + '.bak'
            # Check if we already have a backup; if not, make one
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(file_path, backup_path)
                
            # Save the cleaned data
            df.to_csv(file_path, index=False)
            print(f"Removed {original_shape[0] - new_shape[0]} duplicate rows from {file_path}")
            return original_shape[0] - new_shape[0]
        else:
            print(f"No duplicates found in {file_path}")
            return 0
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def process_trial_folders(base_dir):
    """
    Process all trial folders in the base directory to remove duplicated rows in CSV files.
    """
    # Find all trial folders
    trial_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('trial_')]
    
    if not trial_folders:
        print("No trial folders found.")
        return
    
    print(f"Found {len(trial_folders)} trial folders.")
    
    total_files = 0
    total_duplicates = 0
    
    # Process each trial folder
    for trial_folder in trial_folders:
        trial_path = os.path.join(base_dir, trial_folder)
        
        # Find all word folders
        word_folders = [d for d in os.listdir(trial_path) if os.path.isdir(os.path.join(trial_path, d))]
        
        print(f"Processing trial folder: {trial_folder} - Found {len(word_folders)} word folders.")
        
        # Process each word folder
        for word_folder in word_folders:
            word_path = os.path.join(trial_path, word_folder)
            
            # Find all stage folders
            stage_folders = [d for d in os.listdir(word_path) if os.path.isdir(os.path.join(word_path, d))]
            
            print(f"  Processing word folder: {word_folder} - Found {len(stage_folders)} stage folders.")
            
            # Process each stage folder
            for stage_folder in stage_folders:
                stage_path = os.path.join(word_path, stage_folder)
                
                # Find all CSV files
                csv_files = glob.glob(os.path.join(stage_path, "*.csv"))
                
                print(f"    Processing stage folder: {stage_folder} - Found {len(csv_files)} CSV files.")
                
                # Process each CSV file
                for csv_file in csv_files:
                    print(f"      Processing CSV file: {os.path.basename(csv_file)}")
                    duplicates = remove_duplicates_from_csv(csv_file)
                    total_files += 1
                    total_duplicates += duplicates
    
    print(f"\nProcessing complete. Processed {total_files} files and removed {total_duplicates} duplicate rows.")

if __name__ == "__main__":
    # Determine the base directory
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Try to find Trials_data in the current directory or parent directories
        current_dir = Path.cwd()
        base_dir = None
        
        # Check current directory
        if (current_dir / "Trials_data").exists():
            base_dir = str(current_dir / "Trials_data")
        else:
            # Check up to 3 parent directories
            for i in range(1, 4):
                parent_dir = current_dir.parents[i-1]
                if (parent_dir / "Trials_data").exists():
                    base_dir = str(parent_dir / "Trials_data")
                    break
        
        if base_dir is None:
            print("Could not find Trials_data directory. Please provide the path as an argument.")
            sys.exit(1)
    
    print(f"Processing trial folders in: {base_dir}")
    process_trial_folders(base_dir)