import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from torch._six import inf
import torchmetrics as tm

# return a list of floating point numbers from the string
def get_numbers(s):
    print(s.split('/')[-2])
    newstr = ''.join((ch if ch in '0123456789.' else ' ') for ch in s.split('/')[-2])
    listOfNumbers = [float(i) for i in newstr.split()]
    return listOfNumbers


def metric_scores(target, pred, iou_th):
    device = pred.device

    iou = torch.stack([tm.JaccardIndex(threshold=iou_th, task='binary').to(device)(p, t) for p, t in zip(pred, target)])
    bacc = torch.stack([tm.Accuracy(task='binary').to(device)(p, t) for p, t in zip(pred, target)])
    precision = torch.stack([tm.Precision(task='binary').to(device)(p, t) for p, t in zip(pred, target)])
    recall = torch.stack([tm.Recall(task='binary').to(device)(p, t) for p, t in zip(pred, target)])
    f1s = torch.stack([tm.F1Score(task='binary').to(device)(p, t) for p, t in zip(pred, target)])

    return iou, bacc, precision, recall, f1s

def acc_scores(target, prediction, iou_th):
    target = target
    _, pred = prediction.topk(1, 1, True, True)
    pred = pred.squeeze(1)
    iou, balacc, precision, recall, f1s = metric_scores(target, pred, iou_th=iou_th)
    return iou.mean(), balacc.mean() * 100, precision.mean(), recall.mean(), f1s.mean()

def acc_scores_batch_original(target, prediction, iou_th):
    target = target
    _, pred = prediction.topk(1, 1, True, True)
    pred = pred.squeeze(1)
    iou, balacc, precision, recall, f1s = metric_scores(target, pred, iou_th=iou_th)
    return iou, balacc * 100, precision, recall, f1s

def clip_grad_norm_(parameters, max_norm, i, norm_type=2, do=False):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        if not do:
            print('grad norm', total_norm, i)
        else:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    return total_norm


def save_checkpoint(mlflow, model, optimizer, epoch, losv, iouv, lostr, ioutr):
    state_dict = {"model": model.state_dict(),"optimizer": optimizer.state_dict(),"epoch": epoch,
        "vloss": losv, "viou":iouv, 'trloss':lostr, 'triou':ioutr}
    model_path = f"saved_models/ep{epoch}_model_valiou{iouv}"
    mlflow.pytorch.log_state_dict(state_dict, model_path)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # val is the mean value of a batch, n is the batch size
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("biasqq" not in n) and (p.grad is not None):
            
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            if p.grad.abs().mean() == 0:
                layers.append(n + "ZERO")
            elif p.grad.abs().mean() < 0.00001:
                layers.append(n + "SMALL")
            else:
                layers.append(n)
    
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.2) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return 0


def log_mlflow_metrics(mlflow, type, epoch, iou, acc, prec, rec, f1s, los):
    metrics = {f'{type}.loss':los,f'{type}.iou':iou,f'{type}.balacc':acc,
                f'{type}.precision':prec,f'{type}.recall':rec,f'{type}.f1score':f1s}
    mlflow.log_metrics(metrics, step=epoch)


def log_mlflow_metrics_simplified(mlflow, type, epoch, iou, prec, rec, f1s):
    metrics = {f'{type}.iou':iou, f'{type}.precision':prec,f'{type}.recall':rec,f'{type}.f1score':f1s}
    mlflow.log_metrics(metrics, step=epoch)

def log_mlflow_metrics_iou(mlflow, type, epoch, iou):
    metrics = {f'{type}.iou':iou}
    mlflow.log_metrics(metrics, step=epoch)