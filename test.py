
URL_PATH = "data/malicious_url_scores.csv"

import pandas as pd

# Read the CSV file properly
df = pd.read_csv(URL_PATH)
print(f"Total rows in pandas: {len(df)}")

# Check each column for embedded newlines
for col in df.columns:
    has_newline = df[col].astype(str).str.contains('\n', na=False)
    count = has_newline.sum()
    if count > 0:
        print(f"Column '{col}' has {count} rows with embedded newlines")
        # Show examples
        print("Examples:")
        print(df[has_newline][col].head())
        print()

# Specifically check the URL column
url_with_newlines = df['url'].astype(str).str.contains('\n', na=False)
print(f"\nTotal rows with newlines in URL field: {url_with_newlines.sum()}")

# Now simulate what fgets would see
line_count = 0
with open(URL_PATH, 'r') as f:
    for line in f:
        line_count += 1

print(f"\nRaw line count with fgets: {line_count}")
print(f"Pandas row count (+1 for header): {len(df) + 1}")
print(f"Difference (extra lines from embedded newlines): {line_count - (len(df) + 1)}")

# Find the specific rows with newlines
if url_with_newlines.sum() > 0:
    print("\nRows with embedded newlines:")
    problematic = df[url_with_newlines][['url', 'type']].head(10)
    for idx, row in problematic.iterrows():
        print(f"Row {idx}:")
        print(f"  URL: {repr(row['url'][:100])}")  # repr shows \n explicitly
        print()