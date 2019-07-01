import torch
import torch.nn as nn
import numpy as np


class Classifier(nn.Module):

    def __init__(self, extractor, feat_dim, num_classes=1103, use_focal_loss=True):
        super(Classifier, self).__init__()
        self.extractor = extractor
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(feat_dim, num_classes)
        if use_focal_loss:
            self._init_weights()


    def _init_weights(self):
        def bias_init(prior_prob):
            bias_init = float(-np.log((1 - prior_prob) / prior_prob))
            return bias_init
        nn.init.constant_(self.classifier.bias, bias_init(0.01))
        nn.init.normal_(self.classifier.weight, std=0.01)


    def forward(self, x):
        feat = self.extractor(x)
        pred = self.classifier(feat)
        return pred
