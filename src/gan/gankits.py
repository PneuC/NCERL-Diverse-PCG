import torch
from src.smb.level import MarioLevel
from src.gan.gans import nz
from src.utils.filesys import getpath


def sample_latvec(n=1, device='cpu', distribuion='uniform'):
    if distribuion == 'uniform':
        return torch.rand(n, nz, 1, 1, device=device) * 2 - 1
    elif distribuion == 'normal':
        return torch.randn(n, nz, 1, 1, device=device)
    else:
        raise TypeError(f'unknow noise distribution: {distribuion}')

def process_onehot(raw_tensor_onehot):
    H, W = MarioLevel.height, MarioLevel.seg_width
    res = []
    for single in raw_tensor_onehot:
        data = single[:, :H, :W].detach().cpu().numpy()
        lvl = MarioLevel.from_one_hot_arr(data)
        res.append(lvl)
    return res if len(res) > 1 else res[0]

def get_decoder(path='models/decoder.pth', device='cpu'):
    decoder = torch.load(getpath(path), map_location=device)
    decoder.requires_grad_(False)
    decoder.eval()
    return decoder
    pass


