#!/usr/bin/env python3

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import argparse

CONFIG = {
    "required_columns": ["sample_id"]
}

def validate_test_data(df: pd.DataFrame) -> None:
    required = CONFIG["required_columns"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        available = ", ".join(df.columns[:3])
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Expected columns: {required}\n"
            f"First few columns in input: {available}..."
        )

def load_models(model_root: Path) -> Dict[str, Dict[str, Any]]:
    models = {}
    for model_dir in model_root.glob("*"):
        if model_dir.is_dir():
            model_path = model_dir / "model.pkl"
            metadata_path = model_dir / "metadata.json"
            if model_path.exists() and metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    models[model_dir.name] = {
                        "model": joblib.load(model_path),
                        "metadata": metadata,
                        "features": metadata["features"]
                    }
                except Exception as e:
                    print(f"Failed to load model {model_dir.name}: {str(e)}")
    return models

def preprocess_data(raw_df: pd.DataFrame, features: list) -> pd.DataFrame:
    processed = raw_df[features].map(lambda x: sum(map(int, str(x).split('/'))))
    processed = processed.fillna(processed.mode().iloc[0])
    return processed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_scaler", required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.input)
    validate_test_data(test_df)
    test_df = test_df.set_index("sample_id")

    try:
        scaler = joblib.load(args.global_scaler)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {str(e)}")

    model_store = load_models(Path(args.model_dir))
    print(f"Loaded {len(model_store)} models")

    all_results = []
    for model_id, model_info in tqdm(model_store.items(), desc="Prediction progress"):
        try:
            available_features = set(test_df.columns)
            required_features = set(model_info["features"])
            missing = required_features - available_features
            if missing:
                tqdm.write(f"Model {model_id} missing features: {list(missing)[:3]}... skipped")
                continue
            X_test = test_df[model_info["features"]]
            X_num = preprocess_data(X_test, model_info["features"])
            X_scaled = scaler.transform(X_num)
            preds = model_info["model"].predict(X_scaled)
            result_df = pd.DataFrame({
                "sample_id": X_test.index,
                "predicted_value": np.round(preds, 4)
            })
            output_file = Path(args.output_dir) / f"prediction_{model_id}.csv"
            result_df.to_csv(output_file, index=False)
            all_results.append({
                "model_id": model_id,
                "model_name": model_info["metadata"]["model_name"],
                "samples_predicted": len(preds),
                "output_file": str(output_file)
            })
        except Exception as e:
            tqdm.write(f"Model {model_id} prediction failed: {str(e)}")
            continue

    summary_df = pd.DataFrame(all_results)
    summary_file = Path(args.output_dir) / "prediction_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nPrediction completed! Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
