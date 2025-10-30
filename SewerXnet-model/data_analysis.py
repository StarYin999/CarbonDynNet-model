import os
import pandas as pd
import numpy as np
DATA_DIR = "path"
RECOMMENDED_FIELDS = [
    'Time', 'Date', 'COD', 'NH4', 'SO4', 'H2S', 'pH', 'Q',
    'Flowrate', 'Temperature', 'ORP']
def list_data_files(data_dir=DATA_DIR):
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.xlsx')]
def load_point_data(filepath):
    data = []
    xl = pd.ExcelFile(filepath)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        data.append(df)
    df = pd.concat(data, ignore_index=True)
    return df
def extract_statistics(df, fields=RECOMMENDED_FIELDS):
    stat = {}
    for col in fields:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            stat[col] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'non_na': series.count()
            }
    return stat
def analyze_all_points(data_dir=DATA_DIR):
    files = list_data_files(data_dir)
    all_stats = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            df = load_point_data(f)
            all_stats[name] = extract_statistics(df)
        except Exception as e:
            pass
    return all_stats
if __name__ == '__main__':
    stats = analyze_all_points()
    import json
    print(json.dumps(stats, indent=2, ensure_ascii=False))
