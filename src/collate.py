import torch
from src.features import pad_log_mel

def collate_batch(batch, max_frames=1000):
    features = []
    feature_lengths = []
    targets = []
    target_lengths = []

    for sample in batch:
        log_mel = sample["log_mel"]
        padded, mask = pad_log_mel(log_mel, max_frames)

        features.append(torch.tensor(padded).T)
        feature_lengths.append(int(mask.sum()))

        targets.extend(sample["tokens"])
        target_lengths.append(len(sample["tokens"]))

    features = torch.stack(features)
    targets = torch.tensor(targets, dtype=torch.long)

    return features, targets, feature_lengths, target_lengths
