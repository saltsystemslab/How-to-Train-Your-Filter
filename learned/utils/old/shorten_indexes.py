import pandas as pd
import os
from pathlib import Path

input_folder = Path("data/query_indices")
output_folder = Path("data/compressed_queries")
output_folder.mkdir(exist_ok=True)

for csv_file in input_folder.glob("*.csv"):
    df = pd.read_csv(csv_file)
    
    # Group and count by 'query'
    query_counts = df.groupby("0").size().reset_index(name="count")
    
    # Save result to a new CSV in output folder
    output_file = output_folder / f"{csv_file.stem}.csv"
    query_counts.to_csv(output_file, index=False)
    
    print(f"Processed: {csv_file.name} -> {output_file.name}")