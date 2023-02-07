# -*- coding: utf-8 -*-
"""
Based on this https://www.analyticsvidhya.com/blog/2022/09/dummies-guide-to-writing-a-custom-loss-function-in-tensorflow/

@author: Daniel Barahona
"""
import torch
import torch.nn.functional as F
from torch import nn

# from tensorflow.keras.losses import Loss
# from tensorflow.math import log, multiply_no_nan
# from tensorflow import reduce_mean, reduce_sum

from fuzzylogic.classes import Domain, Set, Rule
from fuzzylogic.hedges import very
from fuzzylogic.functions import R, S, alpha, triangular


# TENSORFLOW VERSION
# class FuzzyLoss(Loss):
#   """
#   Implements Fuzzy Focal Loss as a tuned
#   version of classical Cross-Entropy Loss
#   """

#   def __init__(self, gamma=0):
#     """
#     Args:
#       gamma: Initial value of re-evaluation (default 0 == CE Loss)
#     """
#     super().__init__()
#     self.gamma = gamma

#   def call(self, y_true, y_pred):
#     log_y_pred = log(y_pred) * (1 - y_pred)**self.gamma
#     elements = -multiply_no_nan(x=log_y_pred, y=y_true)
#     return reduce_mean(reduce_sum(elements, axis=1))

# PYTORCH VERSION
class FuzzyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, F.one_hot(target.type(torch.int64)))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy 
        return torch.mean(loss) if self.reduction == 'mean' \
                                else torch.sum(loss) if self.reduction == 'sum' \
                                else loss


def inference():
  # Definir dominios de inputs y output
  balance_deg = Domain("Balance_deg", 0, 1, res=0.1)
  balance_deg.low = S(0.1, 0.5)
  balance_deg.medium = triangular(0.0, 1.0, c=0.5)
  balance_deg.high = R(0.5, 1.0)

  delta_gamma = Domain("Delta_gamma", -0.2, 0.2, res=0.01)
  delta_gamma.ln = S(-0.2, -0.0)
  delta_gamma.ze = triangular(-0.2, 0.2, c=-0.0)
  delta_gamma.lp = R(0.0, 0.2)

  # Reglas
  R1 = Rule({(balance_deg.low,): delta_gamma.ln})
  R2 = Rule({(balance_deg.medium,): delta_gamma.ze})
  R3 = Rule({(balance_deg.high,): very(delta_gamma.lp)})
  rules = R1 | R2 | R3

  # Inferencia
  values = {balance_deg: 0.5}
  print(f' R1: {R1(values)}\n R2: {R2(values)}\n R3: {R3(values)}\n Delta Gamma ===> {rules(values)}')
