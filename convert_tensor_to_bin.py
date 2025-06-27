import torch
from bitarray import bitarray

tensor_file = 'webface10b.t'
bin_file = 'webface10b.bin'
t = torch.load(tensor_file)
t = (t + 1) / 2
t = t.byte()
bits = bitarray(t.flatten().tolist())
with open(bin_file, 'wb') as f:
    bits.tofile(f)
print('Done')
