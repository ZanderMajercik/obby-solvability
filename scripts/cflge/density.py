"""k-NN density estimate and geometric rank sampling for density-mode selection."""
import numpy as np
import torch

# Defaults from the paper supplement ("Density Estimation"): density is the
# negative distance to the k-th nearest reference latent, and the restart slot
# is drawn by a geometric rank sample over candidates sorted most-novel first.
KNN_K_NEIGHBORS = 1000
GEO_SAMPLING_P = 0.005
KNN_MAX_REFERENCE_LATENTS = 25_000  # cap on the reference subset (bounds the cdist matrix)
KNN_CANDIDATE_CHUNK = 4096          # candidates per cdist call (memory bound)


def sample_geometric_with_max(p: float, max_value: int, size=None):
    """Sample from geometric distribution capped at max_value (0-indexed).

    Used by density-mode state selection to bias toward low-density (novel) states.
    """
    if max_value <= 0 or p <= 0:
        return np.zeros(size if size is not None else 1, dtype=np.int64) if size else 0
    sample = np.random.geometric(p, size) - 1
    if size is not None:
        bad = sample > max_value
        while np.any(bad):
            sample[bad] = np.random.geometric(p, int(np.sum(bad))) - 1
            bad = sample > max_value
    elif sample > max_value:
        sample = np.random.randint(0, max_value + 1)
    return sample


def estimate_density(x: torch.Tensor, samples: torch.Tensor,
                     k: int = KNN_K_NEIGHBORS, max_samples: int = 50000,
                     cand_chunk: int = KNN_CANDIDATE_CHUNK) -> torch.Tensor:
    """k-NN density estimate: higher = denser = less novel.

    Processes candidates in chunks of ``cand_chunk`` to keep the cdist matrix
    within a predictable memory budget (cand_chunk × max_samples × 4 bytes).
    """
    if samples.shape[0] > max_samples:
        idx = torch.randint(0, samples.shape[0], (max_samples,), device=samples.device)
        samples = samples[idx]
    k = min(k, samples.shape[0])
    results = []
    for start in range(0, x.shape[0], cand_chunk):
        chunk = x[start:start + cand_chunk]
        kth_dist = torch.cdist(chunk, samples).topk(k, largest=False)[0][:, -1]
        results.append(kth_dist)
    return -torch.cat(results, dim=0)
