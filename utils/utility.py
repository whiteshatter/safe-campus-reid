import os
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
from .nadam import Nadam
from .n_adam import NAdam
import torch.optim.lr_scheduler as lrs

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '':
            if args.save == '': args.save = now
            self.dir = 'experiment/' + args.save
        else:
            self.dir = 'experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = ''
            else:
                self.log = torch.load(self.dir + '/map_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)*args.test_every))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_map_rank(epoch)
        torch.save(self.log, os.path.join(self.dir, 'map_log.pt'))        
        torch.save(
            trainer.feature_center.cpu(), 
            os.path.join(self.dir, 'feature_center.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False, end='\n'):
        print(log, end=end)
        if end != '':
            self.log_file.write(log + end)
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        label = 'Reid on {}'.format(self.args.data_test)
        labels = ['mAP','rank1','rank3','rank5','rank10']
        fig = plt.figure()
        plt.title(label)
        for i in range(len(labels)):
            plt.plot(axis, self.log[:, i].numpy(), label=labels[i])

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('mAP/rank')
        plt.grid(True)
        plt.savefig('{}/test_{}.jpg'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        pass

def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
            }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'amsgrad': args.amsgrad
        }
    elif args.optimizer == 'NADAM':
        optimizer_function = NAdam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler
    
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

