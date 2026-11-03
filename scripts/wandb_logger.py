"""Tiny wandb wrapper.

Used by go_explore_search_solvable.py for optional logging
of encoder-trainer and GoExplore metrics. When ``enabled=False`` (the default),
all calls are no-ops, so the wrapper can be safely embedded in hot paths.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional


class WandbLogger:
    """Thin wrapper around ``wandb`` that becomes a no-op when disabled."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "obby-solvability",
        run_name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        mode: str = "online",
        config: Optional[Mapping] = None,
        group: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self._step = 0
        self.run = None
        if not self.enabled:
            return
        import wandb  # local import so wandb is not required when disabled
        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            tags=list(tags) if tags else None,
            mode=mode,
            config=dict(config) if config is not None else None,
            reinit=True,
        )

    def log(self, data: Mapping, step: Optional[int] = None) -> None:
        # ``step`` is ignored: wandb's own step counter advances, so several call
        # sites (encoder trainer, Go-Explore loop) can log without sharing a step.
        if not self.enabled:
            return
        del step
        self._wandb.log(dict(data))

    def summary(self, **kwargs) -> None:
        if not self.enabled:
            return
        for k, v in kwargs.items():
            self.run.summary[k] = v

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.finish()
        self.run = None
