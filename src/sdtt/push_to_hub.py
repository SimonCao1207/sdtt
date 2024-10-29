import os

import hydra
import lightning as L
from omegaconf import OmegaConf
import torch

from pathlib import Path
from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer

from sdtt.data import utils as dutils
from sdtt.data import dataloader
from sdtt.utils import add_resolvers, prepare_logger, rm_null_values
from sdtt.loading_utils import get_diffusion, get_diffusion_module
from sdtt.run_eval import samples_eval
from lightning.pytorch.loggers import TensorBoardLogger


def get_ckpt_idx(config):
    ckpt_name = config.checkpointing.resume_ckpt_path
    ckpt_idx = ckpt_name.split("/")[-1].split(".")[0]
    ckpt_idx = int(ckpt_idx)
    return ckpt_idx


def push_to_hub(config):
    use_hf_teacher = hasattr(config, "hf")

    if not use_hf_teacher:
        config.parameterization.start_from_hf = False
    else:
        config.parameterization.start_from_hf = True
        print("PUSHING ORIGINAL MDLM TEACHER WEIGHTS")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
    model = get_diffusion(config, tokenizer)

    if not use_hf_teacher:
        assert dutils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        ckpt = torch.load(config.checkpointing.resume_ckpt_path, map_location="cpu")
        ckpt_idx = get_ckpt_idx(config)
        
        state_dict = ckpt["state_dict"]
        # Match expected keys
        new_state_dict = dict()
        for k, v in state_dict.items():
            k = k.replace("backbone.", "")
            k = "backbone." + k
            new_state_dict[k] = v

        state_dict = new_state_dict
        model.load_state_dict(state_dict)
    else:
        ckpt_idx = None

    repo = config.repo
    revision_prefix = config.revision
    revision = f"{revision_prefix}_step_{ckpt_idx}" if ckpt_idx is not None else revision_prefix
    print(f"Pusing to repo: {repo}, on branch: {revision}")
    model.push_to_hub(repo, revision=revision)


"""
How to run the code
python src/sdtt/push_to_hub.py +repo=jdeschena/sdtt +revision=sm_1M_bwd_kl

What do I push?

- baseline mdlm bwd kl distilled
- sm, md, larg at 400k
"""

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    push_to_hub(config)

if __name__ == "__main__":
    add_resolvers()
    prepare_logger()
    main()
