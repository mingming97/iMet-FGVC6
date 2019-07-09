import random
import torch
from torch.utils import data

from dataset import IMetDataset
from dataset.utils.datalist import datalist_from_file
from models import ResNet, ResNeXt, DenseNet, FocalLoss, Classifier
from tools import Trainer
from utils import cfg_from_file
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def parse_args():
    parser = argparse.ArgumentParser(description='IMet FGVC6 ArgumentParser')
    parser.add_argument('--config', default='config/dense121_gc_test.py', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = cfg_from_file(args.config)
    print('using config: {}'.format(args.config))

    data_cfg = cfg['data']
    datalist = datalist_from_file(data_cfg['datalist_path'])
    num_train_files = len(datalist) // 5 * 4
    train_dataset = IMetDataset(data_cfg['dataset_path'],
                                datalist[:num_train_files],
                                transform=data_cfg['train_transform'])
    test_dataset = IMetDataset(data_cfg['dataset_path'],
                               datalist[num_train_files:],
                               transform=data_cfg['test_transform'])
    train_dataloader = data.DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=data_cfg['batch_size'])

    backbone_cfg = cfg['backbone'].copy()
    backbone_type = backbone_cfg.pop('type')
    if backbone_type == 'ResNet':
        backbone = ResNet(**backbone_cfg)
    elif backbone_type == 'ResNeXt':
        backbone = ResNeXt(**backbone_cfg)
    elif backbone_type == 'DenseNet':
        backbone = DenseNet(**backbone_cfg)
    classifier = Classifier(backbone, backbone.out_feat_dim).cuda()

    train_cfg, log_cfg = cfg['train'], cfg['log']
    criterion = FocalLoss().cuda()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=train_cfg['lr'],
                                weight_decay=train_cfg['weight_decay'], momentum=train_cfg['momentum'])
    trainer = Trainer(
        model=classifier, 
        train_dataloader=train_dataloader, 
        val_dataloader=test_dataloader,
        criterion=criterion, 
        optimizer=optimizer,
        train_cfg=train_cfg,
        log_cfg=log_cfg)
    trainer.train()


if __name__ == '__main__':
    main()
