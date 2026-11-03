"""Resolve a method config (configs/**.yaml) into shell variables for the run scripts.

Every program (go_explore_search_solvable.py, infer.py, finetune_ppo.py) takes
only CLI flags. The configs hold those flags in a tree: a config names its
parent with `extends:` and overrides any key of it. configs/README.md shows the
tree.

Keys of a config:
  extends         parent config, relative to this file
  description     one line, for --show
  family          ge | cflge | rl
  variant         output directory name under BASE_LOG_DIR, and the wandb name
  wandb_tags      list of extra wandb tags
  env             variables the run scripts read; a variable set by the caller wins
  flags           program flags: true -> --flag, false/null -> left out,
                  list -> --flag a b, other -> --flag value

Merging: `env` and `flags` merge key by key (the child wins); other keys are
replaced. A value can use ${NAME}, which is taken from the caller's
environment, else from the merged `env`.

Usage:
  python scripts/config.py configs/ge/vanilla.yaml --shell   # export lines, eval'd by the run scripts
  python scripts/config.py configs/ge/vanilla.yaml --show    # chain, env and flags, for reading
  python scripts/config.py --check                           # resolve every config
"""
import argparse
import glob
import os
import re
import shlex
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "configs")
KEYS = {"extends", "description", "family", "variant", "wandb_tags", "env", "flags"}
MERGED = ("env", "flags")


def load(path, seen=()):
    """Config at path merged over its extends chain; also returns the chain."""
    path = os.path.normpath(path)
    if path in seen:
        raise ValueError(f"extends cycle: {' -> '.join(seen + (path,))}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bad = set(cfg) - KEYS
    if bad:
        raise ValueError(f"{path}: unknown keys {sorted(bad)}")
    if "extends" not in cfg:
        return dict(cfg), [path]
    base, chain = load(os.path.join(os.path.dirname(path), cfg["extends"]), seen + (path,))
    for k, v in cfg.items():
        if k in MERGED:
            base[k] = {**base.get(k, {}), **(v or {})}
        elif k != "extends":
            base[k] = v
    return base, chain + [path]


def expand(v, env):
    def sub(m):
        name = m.group(1)
        if name in os.environ:
            return os.environ[name]
        if name in env:
            return str(env[name])
        raise ValueError(f"${{{name}}} is not set")
    return re.sub(r"\$\{(\w+)\}", sub, str(v))


def to_flags(flags, env):
    out = []
    for k, v in flags.items():
        if v is None or v is False:
            continue
        out.append(f"--{k}")
        if v is True:
            continue
        for item in (v if isinstance(v, list) else [v]):
            tok = expand(item, env)
            if not tok or any(c.isspace() for c in tok):
                raise ValueError(f"--{k}: value {tok!r} is empty or has whitespace")
            out.append(tok)
    return out


def resolve(path):
    cfg, chain = load(path)
    flags = cfg.get("flags", {})
    env = {k: os.environ.get(k, str(v)) for k, v in cfg.get("env", {}).items()}
    env = {k: expand(v, env) for k, v in env.items()}
    return cfg, chain, env, to_flags(flags, env)


def shell(path):
    cfg, _, env, flags = resolve(path)
    lines = [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    if "variant" in cfg:
        lines.append(f"export VARIANT={shlex.quote(cfg['variant'])}")
    lines.append(f"export WANDB_TAGS_EXTRA={shlex.quote(' '.join(cfg.get('wandb_tags', [])))}")
    lines.append(f"export METHOD_FLAGS={shlex.quote(' '.join(flags))}")
    return "\n".join(lines)


def show(path):
    cfg, chain, env, flags = resolve(path)
    print("chain:   " + " -> ".join(os.path.relpath(p, ROOT) for p in chain))
    print(f"variant: {cfg.get('variant', '-')}   family: {cfg.get('family', '-')}")
    for k, v in env.items():
        print(f"env:     {k}={v}")
    print("flags:   " + " ".join(flags))


def check():
    """Resolve every config."""
    paths = sorted(glob.glob(os.path.join(CONFIGS, "**", "*.yaml"), recursive=True))
    for p in paths:
        resolve(p)
    print(f"ok: {len(paths)} configs resolve")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shell", action="store_true")
    g.add_argument("--show", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    try:
        if args.check:
            return check()
        if not args.config:
            ap.error("a config path is needed")
        if args.shell:
            print(shell(args.config))
        else:
            show(args.config)
    except (ValueError, OSError, yaml.YAMLError) as e:
        sys.exit(f"config error: {e}")


if __name__ == "__main__":
    main()
