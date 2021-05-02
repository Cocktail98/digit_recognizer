import torch
import struct
import sys
from torchsummary import summary

from model import *

# from utils.torch_utils import select_device

# Initialize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pt_file = './model_weight/best.pt'
# Load model
model = torch.load(pt_file, map_location=device)
# View network architecture and parameters
# net = DigitModel().to(device)
# net.load_state_dict(model)
# summary(net, (1, 28, 28))
# print(model.keys())

with open('../cpp_tensorRT/cnn.wts', 'w') as f:
    f.write('{}\n'.format(len(model.keys())))
    for k, v in model.items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
