import torch
import torch.nn as nn

def train_step(model, batch, optimizer, ctc_loss):
    features, targets, feat_lens, tgt_lens = batch

    optimizer.zero_grad()
    log_probs = model(features)

    loss = ctc_loss(
        log_probs,
        targets,
        torch.tensor(feat_lens),
        torch.tensor(tgt_lens)
    )

    loss.backward()
    optimizer.step()

    return loss.item()
