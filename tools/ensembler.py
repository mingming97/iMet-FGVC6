import os
import torch
from models import ResNet, ResNeXt, DenseNet, Classifier


class Ensembler:

    def __init__(self, net_cfgs, test_dataloader, validate_thresh):
        self.test_dataloader = test_dataloader
        self.validate_thresh = validate_thresh
        self.net_list = []
        for cfg in net_cfgs:
            backbone_cfg = cfg.copy()
            backbone_type = backbone_cfg.pop('type')
            checkpoint = backbone_cfg.pop('checkpoint')

            if backbone_type == 'ResNet':
                backbone = ResNet(**backbone_cfg)
            elif backbone_type == 'ResNeXt':
                backbone = ResNeXt(**backbone_cfg)
            elif backbone_type == 'DenseNet':
                backbone = DenseNet(**backbone_cfg)
            classifier = Classifier(backbone, backbone.out_feat_dim).cuda()

            assert os.path.exists(checkpoint)
            state_dict = torch.load(checkpoint)
            classifier.load_state_dict(state_dict['model_params'])
            classifier.eval()
            self.net_list.append(classifier)


    def inference(self, imgs):
        preds = []
        with torch.no_grad():
            for net in self.net_list:
                pred = net(imgs)
                pred = pred.cpu().sigmoid()
                preds.append(pred)
        res = sum(preds) / len(preds)
        return res


    def test_on_dataloader(self):
        total_predict_positive, total_target_positive, total_true_positive = 0, 0, 0
        print('length of dataloader: {}'.format(len(self.test_dataloader)))
        for data, label in self.test_dataloader:
            data = data.cuda()
            pred = self.inference(data)
            max_pred, _ = pred.max(dim=1, keepdim=True)
            positive_thresh = max_pred * self.validate_thresh
            predict = (pred > positive_thresh).long()
            label = label.type_as(predict)

            total_predict_positive += predict.sum().item()
            total_target_positive += label.sum().item()
            total_true_positive += (label & predict).sum().item()
        p = total_true_positive / total_predict_positive
        r = total_true_positive / total_target_positive
        score = 5 * p * r / (4 * p + r)
        return score
