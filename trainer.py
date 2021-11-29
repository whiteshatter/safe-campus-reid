import os
import os.path
import random
import time
import logging

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
from model.vit.loss import make_loss
from utils.solver import make_optimizer
from utils.solver.scheduler_factory import create_scheduler
from utils.processor import do_train,do_inference
from utils.meter import AverageMeter
from utils.metrics_vit import R1_mAP_eval
from torch.cuda import amp
import torch.nn as nn
from utils.logger import setup_logger
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
        print("=================    the first stage train end    ====================")

        print("=================   the second stage train begin   ====================")
        self.train_lagnet()
        self.model['lagnet'].cpu()
        print("=================    the second stage train end    ====================")

        print("=================   the third stage train begin   ====================")
        self.train_vit()
        self.model['vit'].cpu()
        print("=================    the third stage train end    ====================")

    def test(self):
        print('start test cluster...')
        self.test_cluster(self.model['test'])
        print('test cluster finish')

        print('start test lagnet...')
        self.test_lagnet()
        print('test lagnet finish')

        print('start test vit...')
        self.test_vit()
        print('test vit finish')

    def test_cluster(self, model):
        feature_extractor = model
        store_fs = False
        test_method = 'cosine'
        flips = True
        reranking = False
        result = get_result(feature_extractor, self.loader['cluster']['test']['query'], self.loader['cluster']['test']['gallery'],
                            self.args['cluster']['test'].gpu_ids, store_fs=store_fs, method=test_method, flips=flips, reranking=reranking)
        print(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank5: {:.4f}'.format(
                result['mAP'], result['Top1'], result['Top5']
            ))


    def train_cluster(self):
        train_iterator = self.loader['cluster']['first']['train']
        val_iterator = self.loader['cluster']['first']['val']
        query_iterator = self.loader['cluster']['first']['query']
        gallery_iterator = self.loader['cluster']['first']['gallery']
        id_iterator = self.loader['cluster']['first']['id']
        config = self.config['cluster']['train']
        args = self.args['cluster']['train']
        model = self.model['cluster1']
        self.train_cluster_one_stage(model, config, args, train_iterator, val_iterator, query_iterator, gallery_iterator, id_iterator)
        train_iterator = self.loader['cluster']['second']['train']
        val_iterator = self.loader['cluster']['second']['val']
        query_iterator = self.loader['cluster']['second']['query']
        gallery_iterator = self.loader['cluster']['second']['gallery']
        id_iterator = self.loader['cluster']['second']['id']
        config = self.config['cluster']['train2']
        args = self.args['cluster']['train2']
        model = self.model['cluster2']
        self.train_cluster_one_stage(model, config, args, train_iterator, val_iterator, query_iterator, gallery_iterator, id_iterator)

    def train_cluster_one_stage(self, model, config, args, train_iterator, val_iterator, query_iterator, gallery_iterator, id_iterator):
        optimizer = config.optimizer_func(model)
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
        if args.restore_file is None:
            # move initialization into the model constuctor __init__
            # config.initialization_func(model)
            pass
        else:
            if args.gpu_ids is None:
                checkpoint = torch.load(args.restore_file)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu_ids[0])
                checkpoint = torch.load(args.restore_file, map_location=loc)

            if 'new-optim' in args.version:
                print('==> Reload weights from {}'.format(args.restore_file))
                ckpt = checkpoint
                if 'state_dict' in checkpoint:
                    ckpt = checkpoint['state_dict']
                model.load_state_dict(ckpt)
            else:
                if args.resume_iteration == 0:
                    print('==> Transfer model weights from {}'.format(args.restore_file))
                    if 'external-bnneck' in args.model_name:
                        feature_extractor = model.base
                    else:
                        feature_extractor = model.feature_extractor
                    msg = feature_extractor.load_state_dict(
                        checkpoint, strict=False)
                    print(msg)
                else:
                    print('==> Resume checkpoint {}'.format(args.restore_file))
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    for group in optimizer.param_groups:
                        group['initial_lr'] = args.learning_rate
                        group['lr'] = args.learning_rate
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
                                  id_iterator=id_iterator,
                                  test_params=config.test_params
                                  )
        engine.resume(args.maxepoch, args.resume_epoch, args.resume_iteration)
        model.cpu()
            
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
                print('\r[INFO] epoch: {}\t{}/{}\t{}'.format(
                    epoch, batch + 1, len(train_loader), loss.display_loss(batch)))
            epoch += 1

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

    def train_vit(self):
        cfg = self.config['vit']
        local_rank = 0

        output_dir = cfg.OUTPUT_DIR
        logger = setup_logger("transreid", output_dir, if_train=True)

        set_seed(cfg.SOLVER.SEED)

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        train_loader = self.loader['vit']['train']
        train_loader_normal = self.loader['vit']['train_loader_normal']
        val_loader = self.loader['vit']['val_loader']
        num_query = self.loader['vit']['num_query']
        num_classes = self.loader['vit']['num_classes']
        camera_num = self.loader['vit']['cam_num']
        view_num = self.loader['vit']['view_num']

        model = self.model['vit']

        loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

        optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

        scheduler = create_scheduler(cfg, optimizer)

        # do_train(
        #     cfg,
        #     model,
        #     center_criterion,
        #     train_loader,
        #     val_loader,
        #     optimizer,
        #     optimizer_center,
        #     scheduler,
        #     loss_func,
        #     num_query, local_rank
        # )
        log_period = cfg.SOLVER.LOG_PERIOD
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        eval_period = cfg.SOLVER.EVAL_PERIOD

        device = "cuda"
        epochs = cfg.SOLVER.MAX_EPOCHS

        # logger = logging.getLogger("transreid.train")
        logger.info('start training')
        _LOCAL_PROCESS_GROUP = None
        if device:
            model.to(local_rank)
            if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
                print('Using {} GPUs for training'.format(torch.cuda.device_count()))
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                                  find_unused_parameters=True)

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        scaler = amp.GradScaler()
        # train
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            loss_meter.reset()
            acc_meter.reset()
            evaluator.reset()
            scheduler.step(epoch)
            model.train()
            for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)
                with amp.autocast(enabled=True):
                    score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                    loss = loss_func(score, feat, target, target_cam)

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    scaler.step(optimizer_center)
                    scaler.update()
                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                loss_meter.update(loss.item(), img.shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("[INFO]Epoch[{}] batch:[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                            .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

            if epoch % checkpoint_period == 0:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            if epoch % eval_period == 0:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    def test_vit(self):
        cfg = self.config['vit']
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

        train_loader = self.loader['vit']['train']
        train_loader_normal = self.loader['vit']['train_loader_normal']
        val_loader = self.loader['vit']['val_loader']
        num_query = self.loader['vit']['num_query']
        num_classes = self.loader['vit']['num_classes']
        camera_num = self.loader['vit']['cam_num']
        view_num = self.loader['vit']['view_num']

        model = self.model['vit']
        # model.load_param(cfg.TEST.WEIGHT)

        # do_inference(cfg,
        #              model,
        #              val_loader,
        #              num_query)
        output_dir = cfg.OUTPUT_DIR
        # logger = setup_logger("transreid", output_dir, if_train=True)

        device = "cuda"
        logger = logging.getLogger("transreid.test")
        logger.info("Enter inferencing")

        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()

        if device:
            if torch.cuda.device_count() > 1:
                print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(device)

        model.eval()
        img_path_list = []

        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
            with torch.no_grad():
                img = img.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)
                feat = model(img, cam_label=camids, view_label=target_view)
                evaluator.update((feat, pid, camid))
                img_path_list.extend(imgpath)

        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        print("mAP: {:.1%}, rank1: {:.1%}, rank5: {:.1%}, rank1: {:.10%}".format(mAP, cmc[0], cmc[4], cmc[9]))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


