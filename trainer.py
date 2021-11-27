import os
import os.path

from evaluation import test as get_result
from evaluation import extract_feature as lagnet_extract_feature

# from apex import amp
from model.cluster.construct_engine import construct_engine

import torch
import torch.nn.functional as F
from utils.loss import LAGNetLoss
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking
import utils.utility as utility
import numpy as np
class Trainer():
    def __init__(self, args, config, model, loader):
        self.args = args
        self.config = config
        self.model = model
        self.loader = loader
        self.device = torch.device('cuda')

    def train(self):
        print("=================   the first stage train begin   ====================")
        self.train_cluster()
        self.model['cluster'].cpu()
        print("=================    the first stage train end    ====================")
        
        print("=================   the second stage train begin   ====================")
        self.train_lagnet()
        self.model['lagnet'].cpu()
        print("=================    the second stage train end    ====================")

        pass


    def test(self):
        print('start test cluster...')
        self.test_cluster(self.model['cluster'])
        print('test cluster finish')
        
        print('start test lagnet...')
        self.test_lagnet()
        print('test lagnet finish')

    def test_cluster(self, model):
        feature_extractor = model
        store_fs = False
        test_method = 'cosine'
        flips = True
        reranking = False

        result = get_result(feature_extractor, self.loader['cluster']['query'], self.loader['cluster']['gallery'],
                            self.args['cluster']['test'].gpu_ids, store_fs=store_fs, method=test_method, flips=flips, reranking=reranking)

        for key in result:
            print('{}: {}'.format(key, result[key]))


    def train_cluster(self):
        config = self.config['cluster']['train']
        model = self.model['cluster']
        args = self.args['cluster']['train']
        optimizer = config.optimizer_func(model)
        train_iterator = self.loader['cluster']['train']
        val_iterator = self.loader['cluster']['val']
        query_iterator = self.loader['cluster']['query']
        gallery_iterator = self.loader['cluster']['gallery']
        model = model.cuda()
        if config.lr_scheduler_func:
            lr_scheduler = config.lr_scheduler_func(
                optimizer, **config.lr_scheduler_params)
            lr_scheduler_iter = None
        else:
            lr_scheduler = None
            lr_scheduler_iter = config.lr_scheduler_iter_func(len(), optimizer)

        engine_args = dict(gpu_ids=args.gpu_ids,
                           network=model,
                           criterion=config.loss_func,
                           train_iterator=train_iterator,
                           validate_iterator=val_iterator,
                           optimizer=optimizer,
                           )

        checkpoint_dir = 'static/checkpoint/cluster'
        args.log_dir = os.path.join('logs', 'cluster')
        engine = construct_engine(engine_args,
                                  log_freq=args.log_freq,
                                  log_dir=args.log_dir,
                                  checkpoint_dir=checkpoint_dir,
                                  checkpoint_freq=args.checkpoint_freq,
                                  lr_scheduler=lr_scheduler,
                                  lr_scheduler_iter=lr_scheduler_iter,
                                  metric_dict=config.metric_dict,
                                  query_iterator=query_iterator,
                                  gallary_iterator=gallery_iterator,
                                  id_feature_params=config.id_feature_params,
                                  id_iterator=None,
                                  test_params=config.test_params
                                  )
        engine.resume(args.maxepoch, args.resume_epoch, args.resume_iteration)
            
    def train_lagnet(self):
        args = self.args['lagnet']
        model = self.model['lagnet']
        optimizer = utility.make_optimizer(args, model)
        scheduler = utility.make_scheduler(args, optimizer)
        loss = LAGNetLoss(args)
        train_loader = self.loader['lagnet']['train']
        
        loss.step()
        epoch = scheduler.last_epoch
        lr = scheduler.get_lr()[0]
        loss.start_log()
        model.train()
        feature_center = torch.zeros(args.num_classes, args.num_attentions * args.num_features).to(self.device)
        while epoch <= args.epochs:
            for batch, (inputs, labels, _,_) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                
                feature_center_batch = F.normalize(feature_center[labels], dim=-1)
                feature_center[labels] += args.L2_beta * (outputs[1].detach() - feature_center_batch)  
                loss_ = loss(feature_center_batch, outputs, labels)
                loss_.backward()
                    
                optimizer.step()
            epoch += 1
            
            print('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, args.epochs,
                batch + 1, len(train_loader),
                loss.display_loss(batch)))
            # self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
            #     epoch, self.args.epochs,
            #     batch + 1, len(self.train_loader),
            #     self.loss.display_loss(batch)), 
            # end='' if batch+1 != len(self.train_loader) else '\n')
            loss.end_log(len(train_loader))
        target = model.get_model()
        save_dir = os.path.join('logs', 'lagnet', 'model')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            target.state_dict(), 
            os.path.join(save_dir, 'model_latest.pt')
        )
    
    def test_lagnet(self):
        args = self.args['lagnet']
        test_loader = self.loader['lagnet']['gallery']
        query_loader = self.loader['lagnet']['query']
        testset = self.loader['lagnet']['galleryset']
        queryset = self.loader['lagnet']['queryset']
        model = self.model['lagnet']
        print('\n[INFO] Test:')
        model.eval()

        qf = lagnet_extract_feature(model, query_loader, self.device).numpy()
        gf = lagnet_extract_feature(model, test_loader, self.device).numpy()

        if args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist = cdist(qf, gf)
        r = cmc(dist, queryset.ids, testset.ids, queryset.cameras, testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, queryset.ids, testset.ids, queryset.cameras, testset.cameras)
        print(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
            m_ap,
            r[0], r[2], r[4], r[9]
        ))
        # self.ckpt.write_log(
        #     '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
        #     m_ap,
        #     r[0], r[2], r[4], r[9],
        #     best[0][0],
        #     (best[1][0] + 1)*self.args.test_every
        #     )
        # )