
import torch
torch.set_default_device('cuda')

x = torch.rand(128, 2**10, device='cuda')
out = torch.softmax(x, dim=-1)