import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f


class FocalLoss(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.__num_class = num_class
        self.__alpha = alpha
        self.__gamma = gamma
        self.__smooth = smooth
        self.__size_average = size_average

        if self.__alpha is None:
            self.__alpha = torch.ones(self.__num_class, 1)
        elif isinstance(self.__alpha, (list, np.ndarray)):
            assert len(self.__alpha) == self.__num_class
            self.__alpha = torch.FloatTensor(alpha).view(self.__num_class, 1)
            self.__alpha = self.__alpha / self.__alpha.sum()
        elif isinstance(self.__alpha, float):
            alpha = torch.ones(self.__num_class, 1)
            alpha = alpha * (1 - self.__alpha)
            alpha[balance_index] = self.__alpha
            self.__alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.__smooth is not None:
            if self.__smooth < 0 or self.__smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input_batch, label_batch):
        logit = f.softmax(input_batch, dim=1)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        label_batch = label_batch.view(-1, 1)

        epsilon = 1e-10
        alpha = self.__alpha
        if alpha.device != input_batch.device:
            alpha = alpha.to(input_batch.device)

        idx = label_batch.cpu().long()
        one_hot_key = torch.FloatTensor(label_batch.size(0), self.__num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.__smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.__smooth, 1.0 - self.__smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        log_pt = pt.log()

        gamma = self.__gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * log_pt

        if self.__size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
