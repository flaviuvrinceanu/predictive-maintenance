import pandas as pd

def add_rul(df):
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycle, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df

def cap_rul(df, max_rul):
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df

def get_sensor_columns(df):
    return [col for col in df.columns if col.startswith("s")]

from sklearn.preprocessing import MinMaxScaler

def scale_features(train_df, test_df, feature_cols):
    scaler = MinMaxScaler()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler

def drop_columns(df, cols_to_drop):
    return df.drop(columns=cols_to_drop)