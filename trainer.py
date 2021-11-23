import os
import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from evaluation import test as get_result


# from apex import amp

class Trainer():
    def __init__(self, args, config, model, loader):
        self.args = args
        self.config = config
        self.model = model
        self.loader = loader

    def train(self):
        pass


    def test(self):
        print('start test cluster...')
        self.test_cluster(self.model['cluster'])
        print('test cluster finish')

    def test_cluster(self, model):
        args = self.args['cluster']['test']
        if args.gpu_ids is None:
            checkpoint = torch.load(args.restore_file)
        else:
            loc = 'cuda:{}'.format(args.gpu_ids[0])
            checkpoint = torch.load(args.restore_file, map_location=loc)

        print('==> Resume checkpoint {}'.format(args.restore_file))
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        if 'transfer' in args.version:
            checkpoint = {key: val for key, val in checkpoint.items() if 'classifier' not in key}
            msg = model.load_state_dict(checkpoint, strict=False)
            print(msg)
        else:
            model.load_state_dict(checkpoint)
        feature_extractor = model
        store_fs = False
        test_method = 'cosine'
        flips = True
        reranking = False

        result = get_result(feature_extractor, self.loader['cluster']['query'], self.loader['cluster']['gallery'],
                            self.args['cluster']['test'].gpu_ids, store_fs=store_fs, method=test_method, flips=flips, reranking=reranking)

        for key in result:
            print('{}: {}'.format(key, result[key]))