import numpy as np
import torch

from model_loader import (
    STAGE1_THRESHOLD,
    device,
    fcst15_model,
    fcst30_model,
    fcst60_model,
    stg1_model,
    stg2_model,
)

CLASS_NAMES = ["none", "B", "C", "M", "X"]


def _to_float_list(values):
    return [float(v) for v in values]


@torch.no_grad()
def predict_nowcast(soft, hard, mask):
    soft = np.array(soft, dtype=np.float32)
    hard = np.array(hard, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)

    if soft.shape != (60,) or hard.shape != (60,) or mask.shape != (60,):
        raise ValueError("soft, hard, and mask must each contain exactly 60 values")

    x_soft = torch.tensor(soft[None, :], dtype=torch.float32).to(device)

    stage1_probs = torch.softmax(stg1_model(x_soft), dim=1).cpu().numpy()[0]
    flare_probability = float(stage1_probs[1])
    is_flare = flare_probability >= STAGE1_THRESHOLD

    hard_masked = np.where(mask > 0, hard, 0.0).astype(np.float32)
    stage2_input = np.concatenate([soft, hard_masked, mask], axis=0)
    x_stage2 = torch.tensor(stage2_input[None, :], dtype=torch.float32).to(device)

    stage2_probs = torch.softmax(stg2_model(x_stage2), dim=1).cpu().numpy()[0]
    flare_class_index = int(np.argmax(stage2_probs)) + 1

    if not is_flare:
        flare_class_index = 0

    return {
        "class_index": flare_class_index,
        "class_name": CLASS_NAMES[flare_class_index],
        "flare_probability": flare_probability,
        "threshold": STAGE1_THRESHOLD,
        "stage1": {
            "none": float(stage1_probs[0]),
            "flare": float(stage1_probs[1]),
        },
        "stage2": {
            "B": float(stage2_probs[0]),
            "C": float(stage2_probs[1]),
            "M": float(stage2_probs[2]),
            "X": float(stage2_probs[3]),
        },
    }


@torch.no_grad()
def predict_forecast(soft, hard, mask, horizon):
    soft = np.array(soft, dtype=np.float32)
    hard = np.array(hard, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)

    if soft.shape != (60,) or hard.shape != (60,) or mask.shape != (60,):
        raise ValueError("soft, hard, and mask must each contain exactly 60 values")

    x = np.stack([soft, hard, mask], axis=0)[None, :, :]
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    model = {
        15: fcst15_model,
        30: fcst30_model,
        60: fcst60_model,
    }[horizon]

    probs = torch.sigmoid(model(x_tensor)).cpu().numpy()[0]

    return {
        "horizon_minutes": horizon,
        "any": float(probs[0]),
        "c_or_above": float(probs[1]),
        "m_or_above": float(probs[2]),
    }


def predict_all(soft, hard, mask):
    return {
        "nowcast": predict_nowcast(soft, hard, mask),
        "forecast": {
            "15m": predict_forecast(soft, hard, mask, 15),
            "30m": predict_forecast(soft, hard, mask, 30),
            "60m": predict_forecast(soft, hard, mask, 60),
        },
    }