
import numpy as np

def create_train_sequences(df, feature_cols, seq_length):
    X, y = [], []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")

        for i in range(len(unit_df) - seq_length + 1):
            seq_x = unit_df.iloc[i:i+seq_length][feature_cols].values
            seq_y = unit_df.iloc[i+seq_length-1]["RUL"]

            X.append(seq_x)
            y.append(seq_y)

    return np.array(X), np.array(y)

def create_test_sequences(df, feature_cols, seq_length):
    X = []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        seq = unit_df[feature_cols].values

        if len(seq) >= seq_length:
            seq = seq[-seq_length:]
        else:
            pad_length = seq_length - len(seq)
            padding = np.zeros((pad_length, len(feature_cols)))
            seq = np.vstack([padding, seq])

        X.append(seq)

    return np.array(X)