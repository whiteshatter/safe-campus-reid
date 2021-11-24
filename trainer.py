import os.path

from evaluation import test as get_result


# from apex import amp
from model.cluster.construct_engine import construct_engine


class Trainer():
    def __init__(self, args, config, model, loader):
        self.args = args
        self.config = config
        self.model = model
        self.loader = loader

    def train(self):
        print("=================   the first stage train begin   ====================")
        self.train_cluster()
        self.model['cluster'].cpu()
        print("=================    the first stage train end    ====================")

        pass


    def test(self):
        print('start test cluster...')
        self.test_cluster(self.model['cluster'])
        print('test cluster finish')

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