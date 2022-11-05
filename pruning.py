import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def model_prune(args):
    root = os.path.join('result', args.save_dir, args.pruning_ckt)
    
    checkpoint = torch.load(root)
    model = checkpoint['model']

    encoder_total = 0
    decoder_total = 0
    for name, m in model.named_modules():
        if 'down' in name and isinstance(m, nn.BatchNorm2d):
            encoder_total += m.weight.data.shape[0]
        elif 'up' in name and isinstance(m, nn.BatchNorm2d):
            decoder_total += m.weight.data.shape[0]

    encoder_weight = torch.zeros(encoder_total)
    decoder_weight = torch.zeros(decoder_total)

    idx = 0
    for name, m in model.named_modules():
        if 'down' in name and isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            encoder_weight[idx:(size + idx)] = m.weight.data.abs().clone()
            idx += size

    idx = 0
    for name, m in model.named_modules():
        if 'up' in name and isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            decoder_weight[idx:(size + idx)] = m.weight.data.abs().clone()
            idx += size

    en = encoder_weight.numpy()
    de = decoder_weight.numpy()
    plt.hist(en, bins=200, alpha=0.5)
    plt.hist(de, bins=200, alpha=0.5)
    plt.show()

    ans = np.zeros(encoder_total)
    ans[en <= 0.1] = 1
    print(int(ans.sum()), end='/')
    print(encoder_total)
    ans = np.zeros(decoder_total)
    ans[de<=0.1] = 1
    print(int(ans.sum()), end='/')
    print(decoder_total)
    en_v, i = torch.sort(encoder_weight)
    index = int(encoder_total * args.prune_rate)
    en_threshold = en_v[index]

    de_v, i = torch.sort(decoder_weight)
    index = int(decoder_total * args.prune_rate)
    de_threshold = de_v[index]
    print('encoder average : {:.5f}, max : {:.5f}, threshold : {:.5f}'.format(encoder_weight.mean(), encoder_weight.max(), en_threshold))
    print('decoder average : {:.5f}, max : {:.5f}, threshold : {:.5f}'.format(decoder_weight.mean(), decoder_weight.max(), de_threshold))

    cfg = []
    for name, m in model.named_modules():
        if 'down' in name and isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(en_threshold).float()
            cfg.append(int(torch.sum(mask)))
        elif 'up' in name and isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(de_threshold).float()
            cfg.append(int(torch.sum(mask)))
    cfg[1] = max(cfg[1], cfg[16])
    cfg[16] = max(cfg[1], cfg[16])

    cfg[3] = max(cfg[3], cfg[15])
    cfg[15] = max(cfg[3], cfg[15])

    cfg[6] = max(cfg[6], cfg[14])
    cfg[14] = max(cfg[6], cfg[14])

    cfg[9] = max(cfg[9], cfg[13])
    cfg[13] = max(cfg[9], cfg[13])
    torch.save({'cfg':cfg}, os.path.join('result', args.save_dir, args.pruned_cfg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--config', type=str, default="configs/train_SPKD.yaml", help="Path to yaml config file")
    args = parser.parse_args()

    args = get_config(args.config)
    model_prune(args)