import pandas as pd
from pathlib import Path

# Path to your Parquet file
parquet_path = r"C:\Users\navya\OneDrive\Desktop\FakeNewsDetector\WELFake\train-00000-of-00001-290868f0a36350c5.parquet"

# Set output directory
output_dir = Path("C:/Users/navya/OneDrive/Desktop/FakeNewsDetector/WELFake")

# Create directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Load the Parquet file into a Pandas DataFrame
df = pd.read_parquet(parquet_path)

# Display info about null values and shape
print(df.isnull().sum())
print(df.shape)
df.dropna(inplace=True)
print(df.shape)

# Save CSV file in the specified directory
csv_path = output_dir / "WELFake.csv"
df.to_csv(csv_path, index=False)

print(f"\nCSV file saved at: {csv_path}")