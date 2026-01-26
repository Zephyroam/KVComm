import os
import torch
import argparse
import wandb
import datetime
import logging
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
from models import CVCommunicator
from utils import setup_logging, log_gpu_info, generate_run_name
from dataloader import get_mix_evaluator
from eval_online import CommunicationEvaluator

@dataclass
class AlignConfig:
    # device configuration
    device: str = "cuda:0"
    seed: int = 42
    snapshot_path: str = "snapshots"
    # model configuration
    model_A: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_B: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_input_length: int = 64 * 1000
    # Communication configuration
    layer_from: int = 0
    layer_to: int = 26
    layers_list: list[int] = field(default_factory=lambda: [-1])
    top_layers: float = 0.0
    alpha: float = 1.0
    mu: float = 0.5
    sigma: float = 10.0
    calib_interval: int = 10
    # Test dataset configuration
    test_task: str = "countries_tipsheets"
    mix_method: str = "concat"
    limit: int = 0
    # Test configuration
    do_test: bool = False
    # W&B configuration
    run_name: str = ""
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_tags: str = ""  # comma-separated tags
    # Logging configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

def main(cfg: AlignConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.snapshot_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = generate_run_name(cfg) if cfg.run_name == "" else cfg.run_name
    run_name = f"{run_name}_{timestamp}"

    final_snapshot_path = os.path.join(cfg.snapshot_path, run_name)
    os.makedirs(final_snapshot_path, exist_ok=True)
    log_file_path = os.path.join(final_snapshot_path, "log.log")

    setup_logging(log_file_path=log_file_path, log_level=cfg.log_level)
    logging.info(f"Configuration: {cfg}")
    logging.info(f"All files (logs, models, metrics) will be saved to: {final_snapshot_path}")
    logging.info(f"Log level: {cfg.log_level}")
    log_gpu_info()

    # Initialize W&B
    if cfg.use_wandb:
        wandb_config = {
            k: v for k, v in cfg.__dict__.items() 
            if not k.startswith('wandb_')
        }
        
        wandb_tags = []
        if cfg.wandb_tags != "":
            wandb_tags = [tag.strip() for tag in cfg.wandb_tags.split(',')]
        
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            entity=cfg.wandb_entity,
            tags=wandb_tags,
            config=wandb_config
        )

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_B)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_A = AutoModelForCausalLM.from_pretrained(cfg.model_A, device_map={"": cfg.device}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model_B = AutoModelForCausalLM.from_pretrained(cfg.model_B, device_map={"": cfg.device}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model_A.eval()
    model_B.eval()

    model_A.name = cfg.model_A
    model_B.name = cfg.model_B

    evaluator = get_mix_evaluator(cfg.test_task, cfg.mix_method)
    
    if cfg.limit == 0:
        cfg.limit = None

    results = None
    if cfg.do_test:
        communication_evaluator = CommunicationEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length, cfg)
        cv = CVCommunicator(model_A, model_B, cfg.layer_from, cfg.layer_to, layers_list=cfg.layers_list, top_layers=cfg.top_layers, apply_attn_tracer=True, shift_back=False).to(cfg.device)
        results = communication_evaluator.test(model_A, cv, limit=cfg.limit, calib_interval=cfg.calib_interval)
        
    # Finish W&B run
    if cfg.use_wandb:
        wandb.finish()
    return results


def parse_args() -> AlignConfig:
    parser = argparse.ArgumentParser()
    for field, default in AlignConfig().__dict__.items():
        arg_type = type(default)
        if isinstance(default, bool):
            if default:
                parser.add_argument(f"--no_{field}", dest=field, action="store_false")
            else:
                parser.add_argument(f"--{field}", dest=field, action="store_true")
            parser.set_defaults(**{field: default})
        elif isinstance(default, list):
            element_type = type(default[0])
            parser.add_argument(f"--{field}", type=element_type, default=default, nargs="+")
        else:
            parser.add_argument(f"--{field}", type=arg_type, default=default)
    args = parser.parse_args()
    return AlignConfig(**vars(args))



if __name__ == "__main__":
    config = parse_args()
    main(config)