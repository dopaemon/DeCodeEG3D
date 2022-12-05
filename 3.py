import numpy as np
import pickle
import torch


device = torch.device("cuda")  # TODO
with open("model.pkl", "rb") as f:
    G = pickle.load(f)["G_ema"].cuda()  # torch.nn.Module
