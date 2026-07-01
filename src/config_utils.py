import argparse
import sys
from pathlib import Path

import yaml


DATASET_ALIASES = {
    'INS': 'Instagram',
}


def config_dataset_id(dataset_id):
    base_id = str(dataset_id)
    if base_id.endswith('_dissembled'):
        base_id = base_id[:-len('_dissembled')]
    return DATASET_ALIASES.get(base_id, base_id)


def config_path():
    repo_config = Path(__file__).resolve().parents[1] / 'config.yaml'
    if repo_config.exists():
        return repo_config
    cwd_config = Path('config.yaml')
    if cwd_config.exists():
        return cwd_config
    sys.exit('config.yaml not found')


def load_cfg(dataset_id, mode):
    with config_path().open() as f:
        all_cfg = yaml.safe_load(f)

    cfg_dataset_id = config_dataset_id(dataset_id)
    ds_cfg = all_cfg.get(cfg_dataset_id)
    if ds_cfg is None:
        sys.exit(f'Unknown dataset_id: {dataset_id}')
    mode_cfg = ds_cfg.get(mode)
    if mode_cfg is None:
        sys.exit(f'Unknown mode: {mode} under dataset_id: {cfg_dataset_id}')
    return mode_cfg


def explicit_arg_dests(parser, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    explicit = set()
    for token in argv:
        option = token.split('=', 1)[0]
        for action in parser._actions:
            if option in action.option_strings:
                explicit.add(action.dest)
                break
    return explicit


def apply_cfg(parser, args, cfg, explicit_args=None):
    explicit_args = explicit_args or set()
    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
    }
    for key, value in cfg.items():
        if key in explicit_args:
            continue
        action = action_by_dest.get(key)
        if action is not None and action.type is not None and value is not None:
            value = action.type(value)
        setattr(args, key, value)
