import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRModel(nn.Module):
    """
    Simple BiLSTM-based Acoustic Model for CTC ASR
    """

    def __init__(self, n_mels, hidden_dim, vocab_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        """
        x: (batch, time, n_mels)
        returns: (time, batch, vocab)
        """
        out, _ = self.lstm(x)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=-1)
        return out.transpose(0, 1)
