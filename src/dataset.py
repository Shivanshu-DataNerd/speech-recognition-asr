# src/dataset.py

import os
import pandas as pd

class CommonVoiceAUSDataset:
    """
    Dataset loader for Mozilla Common Voice v24 (English – Australian).
    Supports CSV-based metadata as provided by Data Collective exports.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audio_files")

        # Metadata files
        self.metadata_csv = os.path.join(root_dir, "metadata.csv")

        if not os.path.exists(self.metadata_csv):
            raise FileNotFoundError("metadata.csv not found in dataset directory")

        self.df = self._load_metadata()

    def _load_metadata(self):
        df = pd.read_csv(self.metadata_csv)

        # Basic sanity checks (important for research)
        required_cols = {
            "client_id",
            "path",
            "sentence",
            "accents",
            "duration_ms"
        }

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def __len__(self):
        return len(self.df)

    def get_sample(self, idx: int):
        row = self.df.iloc[idx]

        audio_path = os.path.join(self.audio_dir, row["path"])

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        return {
            "audio_path": audio_path,
            "text": row["sentence"],
            "speaker_id": row["client_id"],
            "accent": row.get("accents"),
            "gender": row.get("gender"),
            "age": row.get("age"),
            "duration_ms": row.get("duration_ms")
        }
