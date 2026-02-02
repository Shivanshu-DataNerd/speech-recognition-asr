import os
import pandas as pd

class CommonVoiceAUSDataset:
    """
    Dataset loader for Mozilla Common Voice v24 (English – Australian).
    """

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: path to commonvoice_en_au directory
        """
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audio_files")
        self.metadata_path = os.path.join(root_dir, "metadata.xlsx")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError("metadata.xlsx not found")

        self.df = pd.read_excel(self.metadata_path)

    def __len__(self):
        return len(self.df)

    def get_sample(self, idx: int):
        row = self.df.iloc[idx]

        audio_path = os.path.join(self.audio_dir, row["path"])
        text = row["sentence"]

        return {
            "audio_path": audio_path,
            "text": text,
            "speaker_id": row["client_id"],
            "accent": row["accents"],
            "gender": row["gender"],
            "age": row["age"],
            "duration_ms": row["duration_ms"]
        }
