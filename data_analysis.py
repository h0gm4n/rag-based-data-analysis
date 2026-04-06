import kagglehub
import polars as pl


# Download dataset:
# path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
# print("Path to dataset files:", path)

data = pl.read_csv("superstore.csv", separator=",", encoding="utf8-lossy")
print(data.shape)
print(data[0])