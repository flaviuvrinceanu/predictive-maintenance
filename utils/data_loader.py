from pathlib import Path
import pandas as pd

COLS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

def load_cmaps_file(file_path):
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLS
    return df