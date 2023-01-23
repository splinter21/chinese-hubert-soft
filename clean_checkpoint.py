import torch

base = torch.load("logs/hubert/9b5mccar/checkpoints/epoch=1-val_acc=0.79.ckpt")

state_dict = base["state_dict"]
new_state_dict = {}

for k, v in state_dict.items():
    if k.startswith("model.") or k.startswith("proj."):
        new_state_dict[k] = v

print(new_state_dict.keys())

torch.save(new_state_dict, "purified-fixed.ckpt")
