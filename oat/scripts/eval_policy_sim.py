"""
Usage:
python experiments/eval_policy_sim.py --checkpoint path/to/ckpt -o path/to/output_dir
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import wandb
import json
import numpy as np
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from typing import List, Optional

@click.command()
@click.option('-c', '--checkpoint', required=True, help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True, help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=1, help="num experiments to run")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('--temperature', default=None, type=float, help="temperature for policy inference")
@click.option('--topk', default=None, type=int, help="topk for policy inference")
@click.option('--use_k_tokens', default=None, type=int, help="number of tokens to use for policy inference")
def eval_policy_sim(
    checkpoint: str,
    output_dir: str,
    num_exp: int = 1,
    device: str = 'cuda:0',
    # policy inference args
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    use_k_tokens: Optional[int] = None,
):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        os.system(f"rm -rf {output_dir}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # grab all checkpoints
    ckpts: List[str]    # file paths to checkpoints to evaluate
    if os.path.isdir(checkpoint):
        ckpts = [
            os.path.join(checkpoint, f) 
            for f in os.listdir(checkpoint) 
            if f.endswith('.ckpt') and f != 'latest.ckpt'
        ]
    else:
        ckpts = [checkpoint,]

    base_output_dir = output_dir
    for ckpt in ckpts:
        # format output dir
        if len(ckpts) > 1:
            ckpt_name = os.path.basename(ckpt).replace('.ckpt', '')
            output_dir = os.path.join(base_output_dir, ckpt_name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir
        
        # load checkpoint
        policy, cfg = BasePolicy.from_checkpoint(ckpt, return_configuration=True)
        
        device = torch.device(device)
        policy.to(device)
        policy.eval()
        
        # run eval
        print(f"Running evaluation on {ckpt}")
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.policy.env_runner,
            output_dir=output_dir,
        )
        
        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if topk is not None:
            kwargs['topk'] = topk
        if use_k_tokens is not None:
            kwargs['use_k_tokens'] = use_k_tokens
        runner_log = env_runner.run(
            policy,
            **kwargs
        )
        
        # Store all runs for computing statistics
        all_runs = []
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                runner_log[key] = [value]
        all_runs.append({k: v for k, v in runner_log.items() if not isinstance(v, list)})
        print(f"Exp 1: success rate = {runner_log['mean_success_rate']}")
        
        for i in range(num_exp - 1):
            this_log = env_runner.run(policy, **kwargs)
            print(f"Exp {i + 2}: success rate = {this_log['mean_success_rate']}")
            all_runs.append({k: v for k, v in this_log.items() if not isinstance(v, list)})
            # merge logs
            for key, value in this_log.items():
                assert key in runner_log
                if isinstance(value, wandb.sdk.data_types.video.Video):
                    runner_log[key].append(value)
                else:
                    runner_log[key] += value
        
        # Compute mean and std for all numeric metrics
        numeric_keys = [k for k in all_runs[0].keys()]
        mean_log = {}
        std_log = {}
        
        for key in numeric_keys:
            values = [run[key] for run in all_runs]
            mean_log[key] = np.mean(values)
            if num_exp > 1:
                std_log[key] = np.std(values, ddof=1)  # sample std
        
        env_runner.close()
        
        # dump log to json
        json_log = dict()
        json_log['checkpoint'] = ckpt
        json_log['num_exp'] = num_exp
        
        # Add mean values
        for key, value in mean_log.items():
            json_log[f'{key}_mean'] = float(value)
        
        # Add standard deviation & error values if multiple experiments
        if num_exp > 1:
            for key, value in std_log.items():
                json_log[f'{key}_std'] = float(value)
                json_log[f'{key}_stderr'] = float(value / np.sqrt(num_exp))
        
        # Add video paths
        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    assert isinstance(video, wandb.sdk.data_types.video.Video)
                    json_log[f'{key}_{i}'] = video._path
        
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    eval_policy_sim()