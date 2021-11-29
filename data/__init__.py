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

def get_cluster2_loader():
    cluster_args = args['cluster']['train2']
    cluster_config = config['cluster']['train2']
    train_loader, eva_loader, query_loader, gallery_loader, id_loader = construct_dataset(cluster_args, cluster_config)

    return train_loader, eva_loader, query_loader, gallery_loader, id_loader

def get_cluster1_loader():
    cluster_args = args['cluster']['train']
    cluster_config = config['cluster']['train']
    train_loader, eva_loader, query_loader, gallery_loader = construct_dataset(cluster_args, cluster_config)

    return train_loader, eva_loader, query_loader, gallery_loader

def get_cluster_test_loader():
    cluster_args = args['cluster']['test']
    cluster_config = config['cluster']['test']
    test_loader = construct_dataset(cluster_args, cluster_config)

    return test_loader

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
    train_loader1, eva_loader1, query_loader1, gallery_loader1 = get_cluster1_loader()
    train_loader2, eva_loader2, query_loader2, gallery_loader2, id_loader = get_cluster2_loader()
    query_loader, gallery_loader = get_cluster_test_loader()
    get_cluster_test_loader()
    loader['cluster'] = {}
    loader['cluster']['first'] = {}
    loader['cluster']['second'] = {}
    loader['cluster']['test'] = {}
    loader['cluster']['first']['train'] = train_loader1
    loader['cluster']['second']['train'] = train_loader2
    loader['cluster']['first']['val'] = eva_loader1
    loader['cluster']['second']['val'] = eva_loader2
    loader['cluster']['first']['query'] = query_loader1
    loader['cluster']['second']['query'] = query_loader2
    loader['cluster']['first']['gallery'] = gallery_loader1
    loader['cluster']['second']['gallery'] = gallery_loader2
    loader['cluster']['first']['id'] = None
    loader['cluster']['second']['id'] = id_loader
    loader['cluster']['test']['query'] = query_loader
    loader['cluster']['test']['gallery'] = gallery_loader
    # #
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


