# creating an albert model
from typing import KeysView
from transformers import AlbertConfig, AlbertModel
import hivemind
from hivemind.optim.grad_averager import GradientAverager
from hivemind.optim.state_averager import TrainingStateAverager
import torch
from functools import partial
import torch_optimizer

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--initial_peers', type=str)
args = parser.parse_args()
print(args.initial_peers)

albert_base_configuration = AlbertConfig(
     hidden_size=768,
     num_attention_heads=12,
     intermediate_size=3072,
)
albert_large_configuration = AlbertConfig(
     hidden_size=1024,
     num_attention_heads=16,
     intermediate_size=4096,
     num_hidden_layers=24
)
albert_xxlarge_configuration = AlbertConfig()


model = AlbertModel(albert_large_configuration)
model.half()
device = torch.device('cuda')
model.to(device)

opt = torch_optimizer.Lamb(model.parameters())
inputs = torch.randint(0,10, [1, 128])
outputs = model(inputs)
outputs = torch.sum(outputs.last_hidden_state)
outputs.backward()
opt.zero_grad()
opt.step()
print(opt.state_dict()['state'][0].keys())

print(f"#param: {sum([param.data.numel() for param in model.parameters()])}")

# distributed environment setting
dht = hivemind.DHT(start=True, host_maddrs=["/ip4/0.0.0.0/tcp/0"], initial_peers=[args.initial_peers])
print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

averager = GradientAverager(
    model.parameters(), dht=dht, prefix="test", target_group_size=2, reuse_grad_buffers=False, start=True
)

control = averager.schedule_step(hivemind.get_dht_time() + 5)

# avgr1 = TrainingStateAverager(
#     dht=dht1, 
#     params=model1.parameters(), 
#     optimizer=partial(torch.optim.Adam, lr=0.1, betas=(0.9, 0.9)), 
#     scheduler=partial(torch.optim.lr_scheduler.LambdaLR, 
#     lr_lambda=lambda t: 1.0 / max(1, t)), 
#     start=True, 
#     prefix="my_exp")
averager.step(control=control, wait=False)
control.result()  # wait for all-reduce to finish