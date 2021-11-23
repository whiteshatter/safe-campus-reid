from datasets import construct_dataset, split_train_val
from config import args, config
from datasets.market1501 import Market1501
from torch.utils.data import dataloader
import torch


def get_cluster_test_loader():
    cluster_args = args['cluster']['train']
    train_transform, val_transform = construct_dataset(args['cluster']['train'], config['cluster']['train'])
    train_all = Market1501(args['cluster']['train'], train_transform, 'train')
    # train_dataset, val_dataset = split_train_val(train_all)
    gallery_dataset = Market1501(args['cluster']['train'], val_transform, 'test')
    query_dataset = Market1501(args['cluster']['train'], val_transform, 'query')

    train_loader = dataloader.DataLoader(train_all, batch_size=cluster_args.batch_size, num_workers=cluster_args.num_workers)
    # eva_loader = dataloader.DataLoader(val_dataset, batch_size=cluster_args.batch_size, num_workers=args.num_workers)
    test_loaders = []
    for test_dataset in [query_dataset, gallery_dataset]:
        test_loaders.append(torch.utils.data.DataLoader(
            test_dataset, batch_size=cluster_args.batch_size, shuffle=False, num_workers=cluster_args.num_workers, pin_memory=True))

    return train_loader, test_loaders[0], test_loaders[1]


def get_loader():
    # 总体loader
    loader = {}

    # cluster相关loader
    train_loader, query_loader, gallery_loader = get_cluster_test_loader()
    loader['cluster'] = {}
    loader['cluster']['train'] = train_loader
    # loader['cluster']['val'] = eva_loader
    loader['cluster']['query'] = query_loader
    loader['cluster']['gallery'] = gallery_loader
    return loader


