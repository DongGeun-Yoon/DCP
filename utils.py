import argparse
import logging
import os

import cv2 as cv
import numpy as np
from numpy.lib.polynomial import poly
import torch
import torch.nn as nn
from scipy.stats import norm

from config import im_size, epsilon, epsilon_sqr, device

from scipy.ndimage import gaussian_filter, morphology
from skimage.measure import label, regionprops

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def safe_crop(mat, x, y, crop_size=(im_size, im_size)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.uint8)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.uint8)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (im_size, im_size):
        ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_NEAREST)
    return ret


# alpha prediction loss: the abosolute difference between the ground truth alpha values and the
# predicted alpha values at each pixel. However, due to the non-differentiable property of
# absolute values, we use the following loss function to approximate it.
def alpha_prediction_loss(y_pred, y_true, mask=None):
    if mask is not None:
        mask = mask
        #diff = y_pred[:, 0, :] - y_true
    else:
        mask = y_true[:, 1, :]
    diff = y_pred[:, 0, :] - y_true[:, 0, :]
    diff = diff * mask
    num_pixels = torch.sum(mask)

    return torch.sum(torch.sqrt(torch.pow(diff, 2) + epsilon_sqr)) / (num_pixels + epsilon)


# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
def compute_mse(pred, alpha, mask):
    num_pixels = mask.sum()
    return ((pred - alpha) ** 2).sum() / num_pixels

# compute the SAD error given a prediction and a ground truth.
def compute_sad(pred, alpha):
    diff = np.abs(pred - alpha)
    return np.sum(diff) / 1000

def compute_grad(pd, gt, mask):
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x ** 2 + pd_y ** 2)
    gt_mag = np.sqrt(gt_x ** 2 + gt_y ** 2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map * mask) / 10
    return loss

# compute the connectivity error
def compute_connectivity(pd, gt, mask, step=0.1):
    h, w = pd.shape

    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]

        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords

        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1

        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i - 1]

        dist_maps = morphology.distance_transform_edt(omega == 0)
        dist_maps = dist_maps / dist_maps.max()
        # lambda_map[flag == 1] = dist_maps.mean()
    l_map[l_map == -1] = 1

    # the definition of lambda is ambiguous
    d_pd = pd - l_map
    d_gt = gt - l_map
    # phi_pd = 1 - lambda_map * d_pd * (d_pd >= 0.15).astype(np.float32)
    # phi_gt = 1 - lambda_map * d_gt * (d_gt >= 0.15).astype(np.float32)
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt) * mask) / 1000
    return loss

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

class DictConfig(object):
    """Creates a Config object from a dict 
       such that object attributes correspond to dict keys.    
    """
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

def get_config(fname):
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config

# ========= #
def spatial_similarity(fm): # spatial similarity
    fm = fm.view(fm.size(0), fm.size(1),-1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001 )
    s = norm_fm.transpose(1,2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def channel_similarity(fm): # channel_similarity
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
    s = norm_fm.bmm(norm_fm.transpose(1,2))
    s = s.unsqueeze(1)
    return s
# ========= #
def fm2vector(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    a = fm.bmm(fm.transpose(1,2))
    sm = a.sum(dim=0)
    sm = sm.sum(dim=0)
    norm_s = (sm - min(sm)) / (max(sm) - min(sm))
    return norm_s

def bn_loss(model, beta, s):
    target = ['down2.conv2.cbr_unit.1', 'down3.conv3.cbr_unit.1', 'down4.conv3.cbr_unit.1', 'down5.conv3.cbr_unit.1']
    b = iter(beta)
    loss = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name in target:
            w = 1 - next(b)
            loss += (m.weight.data * s * w * 10).sum()
        elif isinstance(m, nn.BatchNorm2d):
            loss += (m.weight.data * s).sum()
    return loss

def update_BN(model, s, w=10):
    for name, m in model.named_modules():
        # encoder
        if 'down' in name and isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(torch.sign(m.weight.data)*s*w)
        # decoder
        if 'up' in name and isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(torch.sign(m.weight.data)*s)
