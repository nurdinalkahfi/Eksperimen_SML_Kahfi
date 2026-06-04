import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("../heart.csv")

# Hapus duplikat
df = df.drop_duplicates()

# Pisahkan fitur dan target
X = df.drop("target", axis=1)
y = df["target"]

# Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan hasil preprocessing
preprocessed_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

preprocessed_df["target"] = y.reset_index(drop=True)

preprocessed_df.to_csv(
    "heart_preprocessing.csv",
    index=False
)

print("Preprocessing selesai")
