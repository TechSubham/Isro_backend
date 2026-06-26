import json
from pathlib import Path

import torch

from model_defs import FcstCNN, NBeatsCls

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_model(model, filename):
    path = MODEL_DIR / filename
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


stg1_model = load_state_model(NBeatsCls(L=60, nc=2), "stg1.pt")
stg2_model = load_state_model(NBeatsCls(L=180, nc=4), "stg2_hard.pt")

fcst15_model = load_state_model(FcstCNN(ci=3, no=3), "fcst_cnn15.pt")
fcst30_model = load_state_model(FcstCNN(ci=3, no=3), "fcst_cnn30.pt")
fcst60_model = load_state_model(FcstCNN(ci=3, no=3), "fcst_cnn.pt")

with open(MODEL_DIR / "twostage_thr.json", "r") as f:
    threshold_config = json.load(f)

STAGE1_THRESHOLD = float(threshold_config["thr1"])