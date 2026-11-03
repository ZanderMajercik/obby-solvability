"""Latent encoders for CFLGE density selection.

Each exposes ``encode_target(obs) -> latent``, which GoExplore.compute_world_latents
calls: a frozen dynamics-ckpt encoder, a policy-backbone encoder, and the online
forward-dynamics EncoderTrainer (built by build_encoder_trainer).
"""
import torch


class DynamicsEncoderOnly(torch.nn.Module):
    """Encoder-only mirror of DynamicsModel.

    Architecture:
        LayerNorm(obs_dim) -> Linear(obs_dim, hidden) -> LayerNorm(hidden)
        -> ReLU -> Linear(hidden, latent_dim)

    Loaded from a `DynamicsTrainer.state_dict()` checkpoint (filtering for
    encoder.* keys). predictor + action_embed are not needed for go-explore
    encoding-only use.
    """
    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(obs_dim),
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class DynamicsCkptEncoder(torch.nn.Module):
    """Frozen encoder loaded from a saved DynamicsTrainer checkpoint.

    Exposes the same ``encode_target(obs) -> latent`` API as EncoderTrainer
    so GoExplore.compute_world_latents can call either interchangeably.
    """
    def __init__(self, net, process_obs_fn):
        super().__init__()
        self.net = net
        self._process_obs = process_obs_fn

    def encode_target(self, obs):
        with torch.no_grad():
            out = self._process_obs(*obs)
            if isinstance(out, tuple):
                obs_t, entity_t = out
                x = torch.cat(
                    [obs_t, entity_t.reshape(entity_t.shape[0], -1)], dim=-1)
            else:
                x = out
            return self.net(x.float()).detach()


class BackboneEncoder(torch.nn.Module):
    """Wraps a policy backbone to expose ``encode_target(obs) -> latent``.

    Used when density mode runs with a fixed policy backbone and no
    forward-dynamics trainer (i.e. --density-latent backbone without
    --train-encoder). EncoderTrainer / DynamicsCkptEncoder cover the other
    two cases.
    """
    def __init__(self, action_policy):
        super().__init__()
        self.action_policy = action_policy

    def encode_target(self, obs):
        with torch.no_grad():
            features, _, _ = self.action_policy.backbone([], *obs)
        return features.detach()


def load_dynamics_ckpt_encoder(path, device, verbose=True):
    """Load a frozen encoder from a saved DynamicsTrainer ckpt.

    Returns a module exposing ``encode_target(obs) -> latent`` that can be
    handed to ``GoExplore(encoder=...)``. The ckpt's obs_dim / hidden_dim /
    latent_dim determine the encoder shape; only ``encoder.*`` keys are
    consumed (predictor / action_embed are ignored).
    """
    from madrona_scripts.policy import process_obs as _process_obs_dyn
    sd = torch.load(path, map_location=device, weights_only=False)
    obs_dim_ck = int(sd['obs_dim'])
    latent_dim_ck = int(sd['latent_dim'])
    hidden_dim_ck = int(sd['model']['encoder.1.weight'].shape[0])
    net = DynamicsEncoderOnly(obs_dim_ck, hidden_dim_ck, latent_dim_ck).to(device).eval()
    encoder_sd = {k: v for k, v in sd['model'].items() if k.startswith('encoder.')}
    net.load_state_dict(encoder_sd, strict=True)
    for p in net.parameters():
        p.requires_grad_(False)
    if verbose:
        print(f"[dynamics_ckpt] loaded {path}: "
              f"{obs_dim_ck} -> {hidden_dim_ck} -> {latent_dim_ck} "
              f"(global_step={sd.get('global_step', '?')}, frozen)")
    return DynamicsCkptEncoder(net, _process_obs_dyn)


def build_encoder_trainer(go_explore, *, encoder_init_ckpt=None, encoder_pred_hidden=256,
                           encoder_lr=1e-4, encoder_epochs=2,
                           encoder_minibatch=512, verbose=False, wandb_logger=None):
    """Construct the forward-dynamics EncoderTrainer for a CFLGE go_explore.

    Returns the constructed ``EncoderTrainer``. The caller assigns it to
    ``go_explore.encoder`` (so density-mode encoding goes through it) and,
    if training should fire, wires an on_step callback that drives it.

    Sizing rules:
      * encoder_init_ckpt set → latent_dim comes from ckpt metadata;
        target_obs_dim uses full process_obs output (no kept-cols mask).
      * else if encoder_backbone set → latent_dim from a backbone forward
        pass; target_obs_dim from kept_cols mask (or full process_obs if
        no mask).
      * else (true scratch) → latent_dim = encoder_pred_hidden; full
        process_obs output as target.
    """
    from encoder_trainer import EncoderTrainer
    from madrona_scripts.policy import process_obs as _process_obs_for_target

    if encoder_init_ckpt is not None:
        # latent_dim comes from the checkpoint; EncoderTrainer reads the other
        # layer sizes from it too.
        _ck_meta = torch.load(encoder_init_ckpt, map_location='cpu', weights_only=False)
        latent_dim = int(_ck_meta.get('latent_dim', encoder_pred_hidden))
        del _ck_meta
    elif go_explore.encoder_backbone is not None:
        with torch.no_grad():
            _features, _, _ = go_explore.encoder_backbone.backbone([], *go_explore.obs)
            latent_dim = int(_features.shape[-1])
    else:
        # From scratch (no policy, no checkpoint): latent_dim is
        # encoder_pred_hidden, and the model trains on the full observation.
        latent_dim = int(encoder_pred_hidden)
        print(f"[encoder_trainer] scratch from random init: "
              f"latent_dim={latent_dim} (from --encoder-pred-hidden)")

    with torch.no_grad():
        # The prediction target is the full process_obs output.
        _po = _process_obs_for_target(*go_explore.obs)
        if isinstance(_po, tuple):
            _po_combined = torch.cat(
                [_po[0], _po[1].reshape(_po[1].shape[0], -1)], dim=-1)
        else:
            _po_combined = _po
        _target_obs_dim = int(_po_combined.shape[-1])
        _kept_cols_for_trainer = None

    return EncoderTrainer(
        base_backbone=(go_explore.encoder_backbone.backbone if go_explore.encoder_backbone is not None else None),
        action_dims=go_explore.actions_num_buckets,
        device=go_explore.device,
        latent_dim=latent_dim,
        target_obs_dim=_target_obs_dim,
        process_obs_fn=_process_obs_for_target,
        raw_obs_kept_cols=_kept_cols_for_trainer,
        hidden_dim=encoder_pred_hidden,
        lr=encoder_lr,
        n_epochs=encoder_epochs,
        minibatch_size=encoder_minibatch,
        verbose=verbose,
        wandb_logger=wandb_logger,
        init_from_policy=False,
        init_from_ckpt=encoder_init_ckpt,
    )
