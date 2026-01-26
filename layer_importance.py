import torch
import numpy as np
import logging

def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float32)
    mn, mx = v.min(), v.max()
    if mx - mn < eps:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn + eps)

def gaussian_prior(L, mu, sigma=10.0):
    x = np.arange(L, dtype=np.float32)
    w = np.exp(-0.5 * ((x - mu) / float(sigma)) ** 2)
    return w

def calc_layer_importance(B_attn_weights: dict, model_A_name: str, layer_importance_total: dict):
    layer_importance = {}
    for i in B_attn_weights.keys():
        attn_weights = B_attn_weights[i]
        attn_matrix = attn_weights[..., 1:-attn_weights.shape[-2]]
        importance = attn_matrix.sum(dim=-1).mean()
        layer_importance[i] = 0 if torch.isnan(importance) else importance.item()
        layer_importance_total[i].append(layer_importance[i])
    return layer_importance_total

def get_top_layers(layer_importance_total: dict, cfg):
    topk_layers = get_layer_ranking(layer_importance_total, cfg)
    cfg.layers_list = topk_layers[:int(cfg.top_layers * len(topk_layers))]
    logging.info(f"Top {cfg.top_layers} from {len(topk_layers)} layers: {topk_layers}")
    logging.info(f"New layers list: {cfg.layers_list}")
    return cfg

def get_layer_ranking(layer_importance_total: dict, cfg):
    importance = []
    n_layers = len(layer_importance_total.keys())
    assert n_layers == max(layer_importance_total.keys()) + 1
    for i in range(n_layers):
        importance.append(np.mean(layer_importance_total[i]))
    
    importance = normalize(importance)
    mu = cfg.mu * (n_layers - 1)
    gaussian = gaussian_prior(n_layers, mu=mu, sigma=cfg.sigma)
    gaussian = normalize(gaussian)
    importance = cfg.alpha * importance + (1.0 - cfg.alpha) * gaussian

    top_layers = np.argsort(importance)[::-1]
    logging.info(f"Layer ranking: {top_layers}")
    return top_layers