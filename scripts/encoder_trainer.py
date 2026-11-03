"""Dynamics-model encoder trainer (finetune from a saved DynamicsTrainer ckpt).

The architecture matches the DynamicsTrainer used during PPO training, so a
saved checkpoint can initialize it:

  encoder:    LN(obs_dim) -> Linear(obs_dim, hidden) -> LN(hidden)
              -> ReLU -> Linear(hidden, latent_dim)
  action_emb: Linear(sum(action_buckets), action_emb_dim)
  predictor:  [Linear, LN, ReLU] x predictor_layers -> Linear(hidden, obs_dim)
              fed concat(z, action_emb)
  loss:       MSE(predictor(encode(s_t), action_t), s_{t+1})

obs_dim is the full process_obs output size (e.g. 836), as in the saved
checkpoints.

Used by go_explore_search_solvable.py when --train-encoder is enabled.
With --encoder-init-ckpt PATH, encoder/predictor/action_embed weights are
loaded from a saved DynamicsTrainer state_dict.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _action_onehot(actions: torch.Tensor, action_buckets: Sequence[int]) -> torch.Tensor:
    parts = []
    for i, n in enumerate(action_buckets):
        parts.append(F.one_hot(actions[..., i].long(), num_classes=n))
    return torch.cat(parts, dim=-1).float()


class DynamicsFinetuneModel(nn.Module):
    """Encoder + action-conditioned MLP predictor (matches DynamicsModel)."""

    def __init__(
        self,
        obs_dim: int,
        action_buckets: Sequence[int],
        latent_dim: int = 256,
        hidden_dim: int = 512,
        action_emb_dim: int = 64,
        predictor_layers: int = 3,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_buckets = list(action_buckets)
        self.action_onehot_dim = sum(self.action_buckets)
        self.latent_dim = int(latent_dim)

        self.encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.action_embed = nn.Linear(self.action_onehot_dim, action_emb_dim)

        layers: List[nn.Module] = []
        in_dim = latent_dim + action_emb_dim
        for _ in range(predictor_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, obs_dim))
        self.predictor = nn.Sequential(*layers)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.encode(obs)
        a = self.action_embed(_action_onehot(action, self.action_buckets))
        return self.predictor(torch.cat([z, a], dim=-1))


class EncoderTrainer:
    """Trains DynamicsFinetuneModel by next-state prediction (action-conditioned MSE).

    Go-Explore calls store_obs_t / store_step once per sim step,
    train_minibatches to train, and encode_target to get latents.
    """

    def __init__(
        self,
        base_backbone: nn.Module,                       # unused
        action_dims: Sequence[int],                     # action_buckets
        device: torch.device,
        latent_dim: int,                                # encoder output dim (256)
        target_obs_dim: int,                            # FULL process_obs dim (e.g. 836)
        process_obs_fn,
        raw_obs_kept_cols: torch.Tensor | None = None,  # optional column mask on the targets (see below)
        hidden_dim: int = 512,
        action_emb_dim: int = 64,
        predictor_layers: int = 3,
        lr: float = 1e-4,
        n_epochs: int = 1,
        minibatch_size: int = 4096,
        verbose: bool = False,
        wandb_logger=None,
        init_from_policy: bool = False,                  # unused
        init_from_ckpt: str | None = None,
    ):
        self.process_obs_fn = process_obs_fn
        # raw_obs_kept_cols, if given, selects the target columns the model
        # predicts. Checkpoint-initialized models use the full observation.
        self.raw_obs_kept_cols = raw_obs_kept_cols
        self.target_obs_dim = int(target_obs_dim)
        self.action_buckets = list(action_dims)

        # With init_from_ckpt, the layer sizes come from the checkpoint's
        # tensors, so that load_state_dict matches.
        if init_from_ckpt is not None:
            ck = torch.load(init_from_ckpt, map_location=device, weights_only=False)
            sd = ck['model']
            ck_obs_dim = int(ck.get('obs_dim', self.target_obs_dim))
            ck_latent_dim = int(ck.get('latent_dim', latent_dim))
            ck_hidden_dim = int(sd['encoder.1.weight'].shape[0])
            ck_action_emb_dim = int(sd['action_embed.weight'].shape[0])
            # The predictor's Linear layers are the 2-D 'predictor.N.weight'
            # tensors. The last is the output layer; each earlier one starts a
            # Linear + LN + ReLU block (3 indices per block).
            pred_linear_idx = sorted(
                int(k.split('.')[1]) for k in sd
                if k.startswith('predictor.') and k.endswith('.weight') and sd[k].dim() == 2
            )
            ck_predictor_layers = max(1, len(pred_linear_idx) - 1)
            if ck_obs_dim != self.target_obs_dim:
                raise ValueError(
                    f"ckpt obs_dim={ck_obs_dim} != trainer obs_dim={self.target_obs_dim}"
                )
            self.latent_dim = ck_latent_dim
            hidden_dim = ck_hidden_dim
            action_emb_dim = ck_action_emb_dim
            predictor_layers = ck_predictor_layers
            print(f"[encoder_trainer] ckpt shape: obs_dim={ck_obs_dim} hidden={ck_hidden_dim} "
                  f"latent={ck_latent_dim} act_emb={ck_action_emb_dim} pred_layers={ck_predictor_layers}")
        else:
            self.latent_dim = int(latent_dim)
            ck = None

        self.model = DynamicsFinetuneModel(
            obs_dim=self.target_obs_dim,
            action_buckets=self.action_buckets,
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            action_emb_dim=action_emb_dim,
            predictor_layers=predictor_layers,
        ).to(device).train()

        if ck is not None:
            self.model.load_state_dict(ck['model'], strict=True)
            print(f"[encoder_trainer] init_from_ckpt: loaded {init_from_ckpt} "
                  f"(global_step={ck.get('global_step', '?')})")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.device = device
        self.verbose = verbose
        self.wandb = wandb_logger

        self._obs_t: List[torch.Tensor] | None = None
        self._actions: torch.Tensor | None = None
        self._obs_next: List[torch.Tensor] | None = None

        self.last_loss: float | None = None
        self._train_calls = 0
        self._global_minibatch_step = 0

    # Latent encoding (density-mode selection).
    @torch.no_grad()
    def encode_target(self, obs_in: Sequence[torch.Tensor]) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        x = self._processed_obs(obs_in)
        z = self.model.encode(x)
        if was_training:
            self.model.train()
        return z

    @torch.no_grad()
    def encode_online(self, obs_in: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.encode_target(obs_in)

    # Transition buffer: one sim step of data.
    def store_obs_t(self, obs_in: Sequence[torch.Tensor]) -> None:
        # obs can contain None (the bin_coord_obs slot, used only by PBGE).
        self._obs_t = [o.detach().clone() if o is not None else None for o in obs_in]

    def store_step(self, actions: torch.Tensor, obs_next_in: Sequence[torch.Tensor]) -> None:
        self._actions = actions.detach().clone()
        self._obs_next = [o.detach().clone() if o is not None else None for o in obs_next_in]

    def has_data(self) -> bool:
        return (self._obs_t is not None
                and self._actions is not None
                and self._obs_next is not None)

    def _reset_buffer(self) -> None:
        self._obs_t = None
        self._actions = None
        self._obs_next = None

    def _processed_obs(self, obs_in: Sequence[torch.Tensor]) -> torch.Tensor:
        out = self.process_obs_fn(*obs_in)
        if isinstance(out, tuple):
            obs_t, entity_t = out
            y = torch.cat([obs_t, entity_t.reshape(entity_t.shape[0], -1)], dim=-1)
        else:
            y = out
        if self.raw_obs_kept_cols is not None and y.shape[-1] != self.target_obs_dim:
            y = y.index_select(-1, self.raw_obs_kept_cols)
        return y.float()

    # Training.
    def train_minibatches(self) -> float | None:
        if not self.has_data():
            return None

        N = self._obs_t[0].shape[0]

        last_loss = None
        loss_sum = 0.0
        grad_norm_sum = 0.0
        z_norm_sum = 0.0
        target_norm_sum = 0.0
        n_mb = 0
        for _ in range(self.n_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]
                obs_t_mb = tuple(o[idx] if o is not None else None for o in self._obs_t)
                obs_next_mb = tuple(o[idx] if o is not None else None for o in self._obs_next)
                actions_mb = self._actions[idx]

                x_t = self._processed_obs(obs_t_mb)
                with torch.no_grad():
                    y_target = self._processed_obs(obs_next_mb)
                z = self.model.encode(x_t)
                a_emb = self.model.action_embed(_action_onehot(actions_mb, self.action_buckets))
                y_pred = self.model.predictor(torch.cat([z, a_emb], dim=-1))

                loss = F.mse_loss(y_pred, y_target)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = self._global_grad_norm()
                self.optimizer.step()

                loss_v = float(loss.detach().item())
                last_loss = loss_v
                loss_sum += loss_v
                grad_norm_sum += gnorm
                z_norm_sum += float(z.detach().norm(dim=-1).mean().item())
                target_norm_sum += float(y_target.norm(dim=-1).mean().item())
                n_mb += 1

                if self.wandb is not None:
                    self.wandb.log({
                        "encoder/loss_mb": loss_v,
                        "encoder/grad_norm_mb": gnorm,
                    }, step=self._global_minibatch_step)
                self._global_minibatch_step += 1

        self._train_calls += 1
        self.last_loss = last_loss

        if self.wandb is not None and n_mb > 0:
            self.wandb.log({
                "encoder/loss": loss_sum / n_mb,
                "encoder/grad_norm": grad_norm_sum / n_mb,
                "encoder/z_t_norm": z_norm_sum / n_mb,
                "encoder/y_target_norm": target_norm_sum / n_mb,
                "encoder/train_calls": self._train_calls,
            }, step=self._global_minibatch_step)

        if self.verbose and self._train_calls % 50 == 0:
            print(f"[encoder_trainer] step={self._train_calls} loss={last_loss:.4f}")
        self._reset_buffer()
        return last_loss

    def _global_grad_norm(self) -> float:
        sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        return sq ** 0.5
