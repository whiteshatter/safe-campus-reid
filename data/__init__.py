from datasets import construct_dataset, split_train_val
from config import args, config
from datasets.market1501 import Market1501
from torch.utils.data import dataloader
import torch

from utils.random_erasing import RandomErasing
from data.sampler import RandomSampler
from torchvision import transforms
from importlib import import_module

from datasets.vit.make_dataloader import make_dataloader as get_vit_loader

def get_cluster_test_loader():
    cluster_args = args['cluster']['train']
    train_transform, val_transform = construct_dataset(args['cluster']['train'], config['cluster']['train'])
    train_all = Market1501(args['cluster']['train'], train_transform, 'train')
    train_dataset, val_dataset = split_train_val(train_all)
    gallery_dataset = Market1501(args['cluster']['train'], val_transform, 'test', re_label=False)
    query_dataset = Market1501(args['cluster']['train'], val_transform, 'query', re_label=False)

    train_loader = dataloader.DataLoader(train_dataset, batch_size=cluster_args.batch_size, num_workers=cluster_args.num_workers)
    eva_loader = dataloader.DataLoader(val_dataset, batch_size=cluster_args.batch_size, num_workers=cluster_args.num_workers)
    test_loaders = []
    for test_dataset in [query_dataset, gallery_dataset]:
        test_loaders.append(torch.utils.data.DataLoader(
            test_dataset, batch_size=cluster_args.batch_size, shuffle=False, num_workers=cluster_args.num_workers, pin_memory=True))

    return train_loader, eva_loader, test_loaders[0], test_loaders[1]

def get_lagnet_loader():
    lagnet_arg = args['lagnet']
    train_list = [
        transforms.Resize((lagnet_arg.height, lagnet_arg.width), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if lagnet_arg.random_erasing:
        train_list.append(RandomErasing(probability=lagnet_arg.probability, mean=[0.0, 0.0, 0.0]))

    train_transform = transforms.Compose(train_list)

    test_transform = transforms.Compose([
        transforms.Resize((lagnet_arg.height, lagnet_arg.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not lagnet_arg.test_only:
        module_train = import_module('datasets.' + lagnet_arg.data_train.lower())
        trainset = getattr(module_train, lagnet_arg.data_train)(lagnet_arg, train_transform, 'train')
        train_loader = dataloader.DataLoader(trainset,
                        sampler=RandomSampler(trainset,lagnet_arg.batchid,batch_image=lagnet_arg.batchimage),
                        #shuffle=True,
                        batch_size=lagnet_arg.batchid * lagnet_arg.batchimage,
                        num_workers=lagnet_arg.nThread)
    else:
        train_loader = None
    
    if lagnet_arg.data_test in ['Market1501']:
        module = import_module('datasets.' + lagnet_arg.data_train.lower())
        testset = getattr(module, lagnet_arg.data_test)(lagnet_arg, test_transform, 'test')
        queryset = getattr(module, lagnet_arg.data_test)(lagnet_arg, test_transform, 'query')
    else:
        raise Exception()

    test_loader = dataloader.DataLoader(testset, batch_size=lagnet_arg.batchtest, num_workers=lagnet_arg.nThread)
    query_loader = dataloader.DataLoader(queryset, batch_size=lagnet_arg.batchtest, num_workers=lagnet_arg.nThread)
    return train_loader, test_loader, query_loader, testset, queryset


def get_loader():
    # 总体loader
    loader = {}

    # cluster相关loader
    train_loader, eva_loader, query_loader, gallery_loader = get_cluster_test_loader()
    loader['cluster'] = {}
    loader['cluster']['train'] = train_loader
    loader['cluster']['val'] = eva_loader
    loader['cluster']['query'] = query_loader
    loader['cluster']['gallery'] = gallery_loader

    train_loader, gallery_loader, query_loader, galleryset, queryset = get_lagnet_loader()
    loader['lagnet'] = {}
    loader['lagnet']['train'] = train_loader
    loader['lagnet']['query'] = query_loader
    loader['lagnet']['gallery'] = gallery_loader
    loader['lagnet']['queryset'] = queryset
    loader['lagnet']['galleryset'] = galleryset

    train_loader, train_loader_normal, val_loader, num_query, num_classes, cam_num, view_num = get_vit_loader(config['vit'])
    loader['vit'] = {}
    loader['vit']['train'] = train_loader
    loader['vit']['train_loader_normal'] = train_loader_normal
    loader['vit']['val_loader'] = val_loader
    loader['vit']['num_query'] = num_query
    loader['vit']['num_classes'] = num_classes
    loader['vit']['cam_num'] = cam_num
    loader['vit']['view_num'] = view_num

    return loader


