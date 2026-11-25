"""1-D convolutional neural network used for both spike detection and classification."""
import torch
import torch.nn as nn


class SpikeCNN(nn.Module):
    """Small 1-D CNN that maps a window of the recording to class scores.

    The same architecture is used twice, with a different number of outputs:
      - detector   (n_out=2): "no spike starts here" vs "a spike starts here"
      - classifier (n_out=5): which of the five neuron classes produced the spike

    Design choices:
      - Three conv blocks with kernel size 5 (0.2 ms) pick out local shape
        features such as the rise, peak and undershoot of a spike.
      - Batch normalisation keeps training stable across very different noise levels.
      - AdaptiveAvgPool1d(4) makes the output size independent of the window
        length, so the detector (65 samples) and classifier (105 samples)
        can share the same code.
      - Dropout in the head reduces over-fitting to the single training recording (D1).

    Args:
        n_out: number of output classes.

    Input shape:  (batch, 1, window_length)
    Output shape: (batch, n_out) raw scores (logits); apply softmax for probabilities.
    """

    def __init__(self, n_out: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels, full time resolution
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Block 2: 16 -> 32 channels, then halve the length
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: 32 -> 64 channels, then halve the length again
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Squash whatever length remains down to 4 time steps
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),                  # (batch, 64, 4) -> (batch, 256)
            nn.Dropout(0.3),
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class scores (logits) for a batch of windows."""
        return self.head(self.features(x))
