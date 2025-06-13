import math
from functools import partial
from typing import List, Union, Any

import numpy as np
import torch
import torch.nn as nn
import functools

from torch.nn import ReLU, LeakyReLU, PReLU, Tanh, SELU, ELU


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_func(activation: str):
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation is None:
        return None
    elif activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.2)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "swish":
        return functools.partial(swish)
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def get_loss_func(loss_func: str,lam=0.5) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """
    if loss_func == 'mse':
        return nn.MSELoss()
    elif loss_func == "mse_sum":
        return nn.MSELoss(reduction="sum")
    elif loss_func == "mse_weight":
        return My_MSE_Weighted_Loss()
    elif loss_func == 'mve':
        return nn.GaussianNLLLoss()
    elif loss_func == 'smoothl1':
        return nn.SmoothL1Loss()
    # elif loss_func == 'tanh':
    #     return nn.Tanh()
    # elif loss_func == 'SELU':
    #     return nn.SELU()
    # elif loss_func == 'ELU':
    #     return nn.ELU()
    elif loss_func == "evidential":
        return functools.partial(evidential_loss_new, lam=0.2,epsilon=1e-4)
    elif loss_func == "mse_rank":
        return mse_entropy(lam=lam)
    elif loss_func == "entropy":
        return entropy()

    else:
        raise ValueError(f'Activation "{loss_func}" not supported.')


# updated evidential regression loss
def evidential_loss_new(mu, v, alpha, beta, targets, lam=0.2, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict
    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)
        # + torch.lgamma(alpha) \
        # - torch.lgamma(alpha+0.5)

    L_NLL = torch.mean(nll, dim=-1)  # nll 

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = torch.mean(reg, dim=-1)  # reg 

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG-epsilon)

    return loss



def evidential_loss(mu, v, alpha, beta, targets):
    """
    Use Deep Evidential Regression Sum of Squared Error loss
    :mu: Pred mean parameter for NIG
    :v: Pred lambda parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict
    :return: Loss
    """

    # Calculate SOS
    # Calculate gamma terms in front
    def Gamma(x):
        return torch.exp(torch.lgamma(x))

    coeff_denom = 4 * Gamma(alpha) * v * torch.sqrt(beta)
    coeff_num = Gamma(alpha - 0.5)
    coeff = coeff_num / coeff_denom

    # Calculate target dependent loss
    second_term = 2 * beta * (1 + v)
    second_term += (2 * alpha - 1) * v * torch.pow((targets - mu), 2)
    L_SOS = coeff * second_term

    # Calculate regularizer
    L_REG = torch.pow((targets - mu), 2) * (2 * alpha + v)

    loss_val = L_SOS + L_REG

    return loss_val


class My_MSE_Weighted_Loss(nn.Module):
    def __init__(self):
        super(My_MSE_Weighted_Loss, self).__init__()

    def forward(self, x, y, weight):
        mse_loss = torch.sum(torch.pow((x - y), 2) * weight)
        return mse_loss


class mse_entropy(nn.Module):
    def __init__(self,lam=0.5,epsilon=1e-5):
        super(mse_entropy, self).__init__()

        self.mse = torch.nn.MSELoss()
        self.nll = torch.nn.NLLLoss(reduction='none')
        self.lam = lam
        self.epsilon = epsilon
        self.log = torch.nn.LogSoftmax(dim=-1)


    def forward(self, x, y):

        mse = self.mse(x,y)

        x = x.unsqueeze(dim=-1)
        # y = y.unsqueeze(dim=-1)

        more = torch.exp(x)/(1+torch.exp(x))  # a大于b的概率
        less = 1-more
        x_for_rank = self.log(torch.concat([more,less],dim = -1))

        # x_for_rank = torch.clamp(x_for_rank, min=1e-4, max=1 - 1e-4)

        # label 大于0的部分 为第0类，即A分子活性大于B分子
        y_for_rank = torch.where(y >= 0,
                                 torch.tensor(0).to(device=x.device, dtype=torch.long),
                                 torch.tensor(1).to(device=x.device, dtype=torch.long))

        # loss 权重 将label处于-0.1-0.1的样本的排序loss设为0
        p = torch.where(y > 0.1,
                        torch.tensor(1).to(device=x.device,dtype=torch.float32),
                        y.to(dtype=torch.float32))

        p = torch.where(y < -0.1,
                        torch.tensor(1).to(device=x.device,dtype=torch.float32),
                        p.to(dtype=torch.float32))

        p = torch.where(p  == 1,
                        p,
                        torch.tensor(0).to(device=x.device,dtype=torch.float32)) + self.epsilon

        # nll = torch.mean(self.nll(x_for_rank,y_for_rank) * p) * self.lam 
        nll = torch.mean(self.nll(x_for_rank,y_for_rank)) * self.lam 

        return mse + nll
        # return 2*nll



import torch.nn.functional as F

class entropy(nn.Module):
    def __init__(self):
        super(entropy, self).__init__()


    def forward(self, x, y):


        # label 大于0的部分 为第0类，即A分子活性大于B分子

        y = torch.sign(y)
        x = torch.where(y > 0,
                        x,
                        -x)        # 如果label小于0，需要把预测值取个负号，方便算loss

        y_loss = -F.logsigmoid(x)
        return y_loss.mean()