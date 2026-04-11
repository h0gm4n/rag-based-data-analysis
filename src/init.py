import kagglehub
import os
import shutil

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Check if dataset already exists
data_file = "data/superstore.csv"
if os.path.exists(data_file):
    print(f"Dataset already exists at {data_file}")
else:
    # Download dataset (downloads to cache directory)
    try:
        path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
        print("Downloaded to cache:", path)
        
        # Copy CSV file to data folder
        source_file = os.path.join(path, "superstore.csv")
        if os.path.exists(source_file):
            shutil.copy(source_file, data_file)
            print(f"Copied dataset to {data_file}")
        else:
            print(f"CSV file not found in {path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure the dataset identifier is correct or manually place superstore.csv in the data folder")