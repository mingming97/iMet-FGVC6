from torchvision import transforms

net_cfgs = [
    dict(
    type='ResNet',
    depth=50,
    stage_with_context_block=[False, True, True, True],
    context_block_cfg=dict(ratio=1./4),
    pretrained=False,
    checkpoint='work_dir/res50/res50_gc/checkpoint.pth'),
    dict(
    type='DenseNet',
    depth=121,
    context_block_cfg=dict(
        ratio=1./4),
    pretrained=False,
    checkpoint='work_dir/dense121/dense121_gc/checkpoint.pth'),
    dict(
    type='ResNeXt',
    depth=50,
    stage_with_context_block=[False, True, True, True],
    context_block_cfg=dict(ratio=1./4),
    pretrained=False,
    checkpoint='work_dir/resnext50/resnext50_gc_mixup/checkpoint.pth'),
]

data = dict(
    dataset_path='/home1/liangjianming/imet-2019-fgvc6/train',
    datalist_path='/home1/liangjianming/imet-2019-fgvc6/train.csv',
    batch_size=512,
    test_transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ]),)

validate_thresh = 1./7