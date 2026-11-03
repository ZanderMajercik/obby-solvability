"""Online k-means over archive-slot latents for the cell-free (CFLGE) buffer.

With cell_free on, the slot pool drops the one-slot-per-coarse-bin invariant:
every alive world is inserted each step, and when the pool is full the slots to
evict are the oldest ones in large clusters, so small (rare) clusters survive.
"""
import torch

KMEANS_SLOTS_PER_CENTROID = 50        # default k = pool_size // this (paper: k = 2400 for 120k slots)
KMEANS_INIT_SAMPLES_PER_CENTROID = 10  # k-means++ bootstrap uses this * k latents


def kmeans_pp_init(latents: torch.Tensor, k: int) -> torch.Tensor:
    """k-means++ seeding on latents (N, d). Returns (k, d) centroids."""
    n = latents.shape[0]
    device = latents.device
    x = latents.float()
    idx = torch.randint(0, n, (1,), device=device)
    centroids = [x[idx[0]].clone()]
    min_sq = torch.full((n,), float("inf"), device=device)
    for _ in range(k - 1):
        c = centroids[-1].unsqueeze(0)
        new_sq = torch.cdist(x, c).squeeze(1).pow_(2)
        torch.minimum(min_sq, new_sq, out=min_sq)
        probs = min_sq / (min_sq.sum() + 1e-12)
        idx = torch.multinomial(probs, 1)
        centroids.append(x[idx[0]].clone())
    return torch.stack(centroids)


class CellFreeKMeans:
    """Centroids, cluster sizes and per-slot cluster/age state for the pool.

    The archive (GoExplore) owns the slots and their latents; it passes the live
    mask (``_slot_to_coarse >= 0``) and ``slot_latents`` in where needed.
    """

    def __init__(self, pool_size: int, device, k: int = None,
                 small_quantile: float = 0.2):
        self.pool_size = pool_size
        self.device = device
        self.k_arg = k
        self.small_quantile = float(small_quantile)
        self.k = None
        self.centroids = None        # (k, d) lazy
        self.counts = None           # (k,)   lazy
        self.initialized = False
        self.init_samples = None     # set lazily
        self.global_step = 0
        self.slot_insert_step = None  # (pool_size,) lazy
        self.slot_cluster_id = None   # (pool_size,) lazy

    def lazy_init(self, latent_dim: int) -> None:
        """Allocate state once we know latent_dim."""
        if self.centroids is not None:
            return
        k = self.k_arg if self.k_arg is not None else max(1, self.pool_size // KMEANS_SLOTS_PER_CENTROID)
        self.k = int(k)
        self.init_samples = min(KMEANS_INIT_SAMPLES_PER_CENTROID * self.k, self.pool_size)
        self.centroids = torch.zeros(self.k, latent_dim,
                                     dtype=torch.float32, device=self.device)
        self.counts = torch.zeros(self.k, dtype=torch.int64, device=self.device)
        self.slot_insert_step = torch.zeros(self.pool_size, dtype=torch.int64, device=self.device)
        self.slot_cluster_id = torch.full((self.pool_size,), -1, dtype=torch.int64, device=self.device)

    def initialize_centroids(self, live: torch.Tensor, slot_latents: torch.Tensor) -> None:
        """One-time: seed the centroids with k-means++ and assign cluster ids."""
        vi = torch.where(live)[0]
        latents = slot_latents[vi]
        self.centroids = kmeans_pp_init(latents, self.k)
        ids = torch.cdist(latents.float(), self.centroids).argmin(dim=1)
        self.slot_cluster_id[vi] = ids
        self.counts = torch.bincount(ids, minlength=self.k)
        self.initialized = True

    def nearest(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x.float(), self.centroids).argmin(dim=1)

    def insert(self, latents: torch.Tensor, ids: torch.Tensor) -> None:
        """Running-mean update: add the M new points to their assigned centroids."""
        ones = torch.ones(latents.shape[0], dtype=torch.float32, device=self.device)
        delta = torch.zeros(self.k, dtype=torch.float32, device=self.device)
        delta.scatter_add_(0, ids, ones)
        self.counts += delta.long()
        sum_new = torch.zeros(self.k, latents.shape[-1], dtype=torch.float32, device=self.device)
        sum_new.scatter_add_(0, ids.unsqueeze(1).expand(-1, latents.shape[-1]), latents.float())
        ne = self.counts > 0
        self.centroids[ne] += (
            (sum_new[ne] - delta[ne, None] * self.centroids[ne])
            / self.counts[ne, None].float()
        )

    def remove(self, slots: torch.Tensor, slot_latents: torch.Tensor) -> None:
        """Reverse running-mean update: remove evicted slots from their centroids."""
        ids = self.slot_cluster_id[slots]
        ones = torch.ones(slots.shape[0], dtype=torch.float32, device=self.device)
        delta = torch.zeros(self.k, dtype=torch.float32, device=self.device)
        delta.scatter_add_(0, ids, ones)
        self.counts -= delta.long()
        self.counts.clamp_(min=0)
        sum_ev = torch.zeros(self.k, slot_latents.shape[-1],
                             dtype=torch.float32, device=self.device)
        sum_ev.scatter_add_(0, ids.unsqueeze(1).expand(-1, slot_latents.shape[-1]),
                            slot_latents[slots].float())
        ne = self.counts > 0
        self.centroids[ne] -= (
            (sum_ev[ne] - delta[ne, None] * self.centroids[ne])
            / self.counts[ne, None].float()
        )

    def pick_evictions(self, n_evict: int, live: torch.Tensor) -> torch.Tensor:
        """Pick n_evict slots to evict, by (cluster-large + oldest) rule.

        Returns the slot indices that should be marked free.
        """
        nonempty = self.counts > 0
        ne_counts = self.counts[nonempty].float()
        if ne_counts.numel() == 0:
            # No clusters yet: evict the oldest live slots.
            INF = self.global_step + 1
            score = torch.where(live, self.slot_insert_step,
                                torch.full((self.pool_size,), INF, dtype=torch.int64, device=self.device))
            _, evict_slots = torch.topk(score, n_evict, largest=False)
            return evict_slots

        threshold = torch.quantile(ne_counts, self.small_quantile)
        large_mask = self.counts > threshold
        in_large = large_mask[self.slot_cluster_id.clamp(min=0)]
        eligible = live & in_large
        n_cand = int(eligible.sum().item())

        if n_cand < n_evict:
            n_protect = int(self.small_quantile * int(nonempty.sum().item()))
            sorted_c = torch.argsort(self.counts)
            protected = sorted_c[:n_protect]
            sm = torch.zeros(self.k, dtype=torch.bool, device=self.device)
            sm[protected] = True
            large_mask = ~sm & nonempty
            in_large = large_mask[self.slot_cluster_id.clamp(min=0)]
            eligible = live & in_large

        INF = self.global_step + 1
        score = torch.where(eligible, self.slot_insert_step,
                            torch.full((self.pool_size,), INF, dtype=torch.int64, device=self.device))
        _, evict_slots = torch.topk(score, n_evict, largest=False)
        return evict_slots
