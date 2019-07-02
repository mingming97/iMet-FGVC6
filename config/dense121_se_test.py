from torchvision import transforms

backbone = dict(
    type='DenseNet',
    depth=121,
    context_block_cfg=dict(
        ratio=1./16, 
        context_modeling_type='avg',
        fusion_type='mul'),
    pretrained=True,)

data = dict(
    dataset_path='/home1/liangjianming/imet-2019-fgvc6/train',
    datalist_path='/home1/liangjianming/imet-2019-fgvc6/train.csv',
    batch_size=32,
    train_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ]),
    test_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ]),)

train = dict(
    epoch=300,
    lr=0.01,
    weight_decay=0.0001,
    momentum=0.9,
    lr_cfg=dict(
        gamma=0.1,
        step=[200, 250]),
    validate_thresh=1/7,
    accumulate_batch_size=256,
    checkpoint=None,)


log = dict(
    log_dir='./log',
    log_file='res50.log',
    print_frequency=20,)

