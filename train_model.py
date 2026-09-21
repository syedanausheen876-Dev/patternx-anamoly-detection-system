import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = "CICIDS2017_cleaned.csv"

print("=" * 70)
print("PATTERNX - RANDOM FOREST MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(FILE_PATH)

print("Dataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nChecking missing values...")

missing = df.isnull().sum().sum()

print("Total missing values:", missing)

if missing > 0:
    print("Removing rows containing missing values...")
    df = df.dropna()


# ============================================================
# 3. TARGET COLUMN
# ============================================================

TARGET = "Attack Type"

if TARGET not in df.columns:
    raise ValueError("Attack Type column not found!")

print("\nTarget column:", TARGET)

print("\nAttack classes:")
print(df[TARGET].value_counts())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nNumber of input features:", X.shape[1])

print("\nFeatures used by PatternX:")

for i, column in enumerate(X.columns, 1):
    print(f"{i}. {column}")


# ============================================================
# 5. HANDLE INFINITE VALUES
# ============================================================

print("\nChecking infinite values...")

X = X.replace([float("inf"), float("-inf")], 0)

X = X.fillna(0)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset into training and testing data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest model...")
print("Please wait. Large dataset may take some time.")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================================
# 8. TEST MODEL
# ============================================================

print("\nTesting model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print("\nTop important features:")

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance.head(15))


# ============================================================
# 11. SAVE MODEL
# ============================================================

print("\nSaving PatternX model...")

joblib.dump(model, "patternx_model.pkl")

joblib.dump(list(X.columns), "patternx_features.pkl")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PATTERNX MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nDataset rows       :", len(df))
print("Input features    :", X.shape[1])
print("Training rows     :", len(X_train))
print("Testing rows      :", len(X_test))
print(f"Test Accuracy     : {accuracy * 100:.2f}%")

print("\nSaved files:")
print("1. patternx_model.pkl")
print("2. patternx_features.pkl")

print("\nPatternX Random Forest model is ready.")