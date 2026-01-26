
from datasets import concatenate_datasets


def mix_iid(ds_a, ds_b, random_state=42):
    ds = mix_concat(ds_a, ds_b)
    ds = ds.shuffle(seed=random_state)
    return ds

def mix_concat(ds_a, ds_b):
    return concatenate_datasets([ds_a, ds_b])


def mix_datasets(ds_a, ds_b, mix_method="concat", random_state=42):
    if mix_method == "concat":
        return mix_concat(ds_a, ds_b)
    elif mix_method == "iid":
        return mix_iid(ds_a, ds_b, random_state=random_state)
    else:
        raise ValueError(f"Unknown mix method: {mix_method}")