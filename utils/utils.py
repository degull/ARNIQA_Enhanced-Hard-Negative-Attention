import argparse
import yaml
from dotmap import DotMap
from functools import reduce
from operator import getitem
from distutils.util import strtobool
from pathlib import Path
import torch

from scipy import stats

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

def calculate_srcc_plcc(proj_A, proj_B):
    """SRCC와 PLCC 계산"""
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


def save_checkpoint(model, checkpoint_path, epoch, srocc):
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

def parse_config(config_file_path: str) -> DotMap:
    """YAML 설정 파일을 파싱합니다."""
    with open(config_file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return DotMap(config, _dynamic=False)

def parse_command_line_args(config: DotMap) -> DotMap:
    """커맨드 라인 인수를 파싱합니다."""
    parser = argparse.ArgumentParser()

    def add_arguments(section, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                add_arguments(value, prefix=full_key)
            else:
                if isinstance(value, list):
                    parser.add_argument(f'--{full_key}', default=value, type=type(value[0]), nargs='+', help=f'{full_key}의 값')
                else:
                    if type(value) == bool:
                        parser.add_argument(f'--{full_key}', default=value, type=strtobool, help=f'{full_key}의 값')
                    else:
                        parser.add_argument(f'--{full_key}', default=value, type=type(value), help=f'{full_key}의 값')

    add_arguments(config)

    args, _ = parser.parse_known_args()
    args = DotMap(vars(args), _dynamic=False)
    return args

def merge_configs(config: DotMap, args: DotMap) -> DotMap:
    """커맨드 라인 인수를 설정에 병합합니다. 커맨드 라인 인수는 설정 파일보다 우선합니다."""
    
    def update_config(config, key, value):
        *keys, last_key = key.split('.')
        reduce(getitem, keys, config)[last_key] = value

    def get_updates(config, args, prefix=""):
        keys_to_modify = []
        
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                if not hasattr(args, key):
                    setattr(args, key, DotMap())
                keys_to_modify.extend(get_updates(value, getattr(args, key), prefix=full_key))
            else:
                if hasattr(args, key):
                    keys_to_modify.append((full_key, getattr(args, key)))
                else:
                    setattr(args, key, value)
        
        return keys_to_modify

    keys_to_modify = get_updates(config, args)

    for key, value in keys_to_modify:
        update_config(config, key, value)

    return config
