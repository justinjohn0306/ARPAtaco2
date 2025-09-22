import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    """
    lengths: 1D LongTensor of shape (batch,)
    returns: mask of shape (batch, max_len) dtype=bool
    """
    # max_len on the same device as lengths
    max_len = int(torch.max(lengths).item())
    # create arange on same device & dtype
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)  # shape (1, max_len)
    mask = (ids < lengths.unsqueeze(1))  # broadcasting -> (batch, max_len)
    return mask.bool()


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    # ensure float32 tensor
    return torch.from_numpy(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    """
    Move tensor to GPU if available. Keeps pinning/non_blocking behavior if caller uses pinned memory.
    Returns the tensor (no Variable wrapper).
    """
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x
