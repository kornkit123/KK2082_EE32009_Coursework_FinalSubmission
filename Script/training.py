"""Tensor conversion, noise augmentation and the training loop for SpikeCNN."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, DEVICE


def to_tensor(windows, mu, sigma):
    """Normalise windows and convert them to the (batch, 1, length) tensor the CNN expects.

    The same (mu, sigma) must be used for training and prediction, otherwise
    the network sees inputs on a different scale from the ones it learned on.

    Args:
        windows: 2-D array (n_windows, window_length).
        mu:      value subtracted from every sample.
        sigma:   value every sample is divided by.

    Returns:
        float32 tensor of shape (n_windows, 1, window_length), on the CPU.
    """
    return torch.tensor(((windows - mu) / sigma)[:, None, :], dtype=torch.float32)


def make_loader(windows, labels, shuffle, mu, sigma, batch_size=BATCH_SIZE):
    """Wrap windows and integer labels in a PyTorch DataLoader.

    Args:
        windows:    2-D array (n_windows, window_length).
        labels:     1-D array of integer class labels, one per window.
        shuffle:    True for training data, False for validation data.
        mu, sigma:  normalisation constants (see to_tensor).
        batch_size: windows per batch.

    Returns:
        DataLoader yielding (x, y) batches.
    """
    x = to_tensor(windows, mu, sigma)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def add_noise(xb, levels_extra, weights=None):
    """Add fresh Gaussian noise at a randomly chosen level to each window in a batch.

    Called on every training batch, so the network never sees the same noise
    twice. This lets a model trained only on clean D1 generalise to D2-D6.

    Args:
        xb:           batch tensor (batch, 1, length), already normalised.
        levels_extra: 1-D tensor of extra noise std per level (normalised units).
        weights:      optional 1-D tensor, one sampling weight per level
                      (need not sum to 1). None means all levels are equally likely.

    Returns:
        Tensor of the same shape as `xb` with noise added.
    """
    if weights is None:
        idx = torch.randint(len(levels_extra), (len(xb),), device=xb.device)
    else:
        idx = torch.multinomial(weights, len(xb), replacement=True)
    sigma = levels_extra[idx].view(-1, 1, 1)     # one noise level per window
    return xb + sigma * torch.randn_like(xb)


def train_model(model, train_dl, val_dl, loss_fn, epochs=40, lr=1e-3, patience=8,
                augment=None, device=DEVICE, verbose=True):
    """Train a model with early stopping and return the best weights found.

    Each epoch runs one pass over the training data (with optional augmentation)
    and one over the validation data. The learning rate is halved when the
    validation loss stops improving, and training stops after `patience`
    epochs without improvement.

    Args:
        model:    the network to train (modified in place).
        train_dl: DataLoader of training batches.
        val_dl:   DataLoader of validation batches.
        loss_fn:  loss function, e.g. nn.CrossEntropyLoss().
        epochs:   maximum number of epochs.
        lr:       initial learning rate for AdamW.
        patience: epochs without validation improvement before stopping.
        augment:  optional function applied to each training batch (e.g. add_noise).
        device:   'cuda' or 'cpu'.
        verbose:  print the validation loss and accuracy after every epoch.

    Returns:
        The model, loaded with the weights that gave the lowest validation loss.
    """
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=3)

    best_loss, best_state, epochs_no_improve = np.inf, None, 0

    for epoch in range(epochs):
        # ---- training pass ----
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            if augment is not None:
                xb = augment(xb)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()

        # ---- validation pass ----
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += loss_fn(out, yb).item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
        n = len(val_dl.dataset)
        val_loss /= n
        scheduler.step(val_loss)
        if verbose:
            print(f'epoch {epoch + 1:2d} | val loss {val_loss:.4f} | val acc {correct / n:.3f}')

        # ---- early stopping: remember the best weights ----
        if val_loss < best_loss - 1e-4:
            best_loss, epochs_no_improve = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:          # None only if epochs == 0
        model.load_state_dict(best_state)
    return model


def accuracy(model, windows, labels, mu, sigma, device=DEVICE):
    """Fraction of windows the model labels correctly.

    Args:
        model:     trained network.
        windows:   2-D array (n_windows, window_length).
        labels:    1-D array of true integer labels.
        mu, sigma: normalisation constants (see to_tensor).
        device:    'cuda' or 'cpu'.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    model.eval()
    with torch.no_grad():
        predicted = model(to_tensor(windows, mu, sigma).to(device)).argmax(1).cpu().numpy()
    return (predicted == labels).mean()
