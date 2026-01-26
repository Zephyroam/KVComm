import os
import logging
import torch

def log_gpu_info():
    if torch.cuda.is_available():
        logging.info(f"Number of GPU: {torch.cuda.device_count()}")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
    else:
        logging.warning("CUDA is not available!")

def setup_logging(log_file_path: str, log_level: str = "INFO"):
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    level = log_level_map.get(log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, encoding='utf-8')
        ]
    )
    
    logging.info(f"Logging setup completed. Log file: {log_file_path}")
    logging.info(f"Log level: {log_level}")

def generate_run_name(cfg) -> str:
    def get_model_short_name(model_path: str) -> str:
        name = model_path.split("/")[-1]
        return name.replace("-", "").replace("_", "").lower()
    
    model_A_short = get_model_short_name(cfg.model_A)
    model_B_short = get_model_short_name(cfg.model_B)
    
    base_name = f"{model_A_short}-to-{model_B_short}"
    
    if cfg.top_layers > 0:
        layer_info = f"top{cfg.top_layers}"
    elif cfg.layers_list != [-1]:
        layer_info = f"layers{cfg.layers_list}"
    else:
        layer_info = f"from{cfg.layer_from}to{cfg.layer_to}"
    base_name += f"_{layer_info}"
    
    return base_name

def generate_run_name_multi_agent(cfg) -> str:
    def get_model_short_name(model_path: str) -> str:
        name = model_path.split("/")[-1]
        return name.replace("-", "").replace("_", "").lower()
    
    model_A1_short = get_model_short_name(cfg.model_A1)
    model_A2_short = get_model_short_name(cfg.model_A2)
    model_B_short = get_model_short_name(cfg.model_B)
    
    base_name = f"{model_A1_short}-{model_A2_short}-to-{model_B_short}"
    
    if cfg.top_layers > 0:
        layer_info = f"top{cfg.top_layers}"
    elif cfg.layers_list != [-1]:
        layer_info = f"layers{cfg.layers_list}"
    else:
        layer_info = f"from{cfg.layer_from}to{cfg.layer_to}"
    base_name += f"_{layer_info}"
    
    return base_name