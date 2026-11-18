"""Download the real Boston Housing dataset into data/boston.csv (needs internet)."""
from src.data_loader import download_from_openml
from src.config import DATA_PATH
import os

df = download_from_openml()
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
df.to_csv(DATA_PATH, index=False)
print(f"Saved {len(df)} rows to {DATA_PATH}")
print("Now run:  python train.py")
