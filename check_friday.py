import pandas as pd

file = "Friday.csv"

print("Reading Friday.csv...")
df = pd.read_csv(file, nrows=5)

print("\n========== DATASET INFORMATION ==========")

# Total rows in full dataset
total_rows = sum(1 for _ in open(file, encoding="utf-8", errors="ignore")) - 1

print("Rows       :", total_rows)
print("Columns    :", len(df.columns))

print("\n========== COLUMN NAMES ==========")

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

print("\n========== IMPORTANT COLUMNS CHECK ==========")

keywords = [
    "ip",
    "source",
    "destination",
    "host",
    "port",
    "timestamp",
    "attack",
    "label"
]

for col in df.columns:
    col_lower = str(col).lower()

    if any(word in col_lower for word in keywords):
        print("FOUND:", col)

print("\n========== FIRST 2 ROWS ==========")
print(df.head(2).to_string())

print("\n========== DONE ==========")