import os
import torch
import argparse
import wandb
import datetime
import logging
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
from models_ms import CVCommunicator
from models_cipher import CipherAgent
from typing import Literal
from utils import setup_logging, log_gpu_info, generate_run_name_multi_agent
from dataloader import get_multi_agent_evaluator
from eval_ms import CommunicationEvaluator, NLDEvaluator, CipherEvaluator
from layer_importance import get_top_layers, get_layer_ranking
import random

@dataclass
class AlignConfig:
    # device configuration
    device: str = "cuda:0"
    seed: int = 42
    snapshot_path: str = "snapshots"
    # model configuration
    model_A1: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_A2: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_B: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_input_length: int = 64 * 1000
    # Communication configuration
    layer_from: int = 0
    layer_to: int = 26
    layers_list: list[int] = field(default_factory=lambda: [-1])
    top_layers: float = 0.0
    calib_size: int = 1
    alpha: float = 1.0
    mu: float = 0.5
    sigma: float = 10.0
    random_selection: bool = False
    # Test dataset configuration
    test_task: str = "tipsheets"
    task_name: str = ""
    limit: int = 0
    # Test configuration
    do_test: bool = False
    do_test_nld: bool = False
    do_test_cipher: bool = False
    # NLD configuration
    # max tokens to generate for model A and B in phase 1
    nld_max_tokens_model_A_and_B_phase1: int = 128
    sender_aware: bool = False
    # AC configuration
    f: Literal["replace", "sum", "mean"] = "replace"
    layer_k: int = 26
    layer_j: int = 26
    # W&B configuration
    run_name: str = ""
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_tags: str = ""  # comma-separated tags
    # Logging configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

def prepare_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    model.name = model_name
    # special case for Gemma
    if "gemma" in model_name.lower():
        torch._dynamo.config.cache_size_limit = 64
    return model

def main(cfg: AlignConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.snapshot_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = generate_run_name_multi_agent(cfg) if cfg.run_name == "" else cfg.run_name
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

    logging.info(f"Multi sender using model_A1: {cfg.model_A1}, model_A2: {cfg.model_A2}, and model_B: {cfg.model_B}")
    model_A1 = prepare_model(cfg.model_A1, cfg.device)
    model_A2 = prepare_model(cfg.model_A2, cfg.device)
    model_B = prepare_model(cfg.model_B, cfg.device)

    evaluator = get_multi_agent_evaluator(cfg.test_task)
    
    if cfg.limit == 0:
        cfg.limit = None

    results = None
    if cfg.do_test:
        communication_evaluator = CommunicationEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length)
        if cfg.top_layers > 0:
            cv = CVCommunicator(model_A1, model_A2, model_B, cfg.layer_from, cfg.layer_to, layers_list=cfg.layers_list, top_layers=cfg.top_layers, apply_attn_tracer=True).to(cfg.device)
            if cfg.random_selection:
                cfg.layers_list = random.sample(list(range(0, cv.A_num_layers)), int(cfg.top_layers * cv.A_num_layers))
                logging.info(f"Randomly selected layers list: {cfg.layers_list}")
            else:
                communication_evaluator.test(model_A1, model_A2, cv, limit=cfg.calib_size, no_wandb=True, do_calc_layer_importance=True)
                cfg = get_top_layers(communication_evaluator.layer_importance_total, cfg)
        
        cv = CVCommunicator(model_A1, model_A2, model_B, cfg.layer_from, cfg.layer_to, layers_list=cfg.layers_list, top_layers=cfg.top_layers, apply_attn_tracer=False).to(cfg.device)
        results = communication_evaluator.test(model_A1, model_A2, cv, limit=cfg.limit)
    if cfg.do_test_nld:
        nld_evaluator = NLDEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length, cfg.nld_max_tokens_model_A_and_B_phase1, cfg.sender_aware)
        results = nld_evaluator.test(model_A1, model_A2, model_B, limit=cfg.limit)
    if cfg.do_test_cipher:
        model_A1 = CipherAgent(model_A1, tokenizer)
        model_A2 = CipherAgent(model_A2, tokenizer)
        model_B = CipherAgent(model_B, tokenizer)
        cipher_evaluator = CipherEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length, cfg.nld_max_tokens_model_A_and_B_phase1, cfg.sender_aware)
        results = cipher_evaluator.test(model_A1, model_A2, model_B, limit=cfg.limit)
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