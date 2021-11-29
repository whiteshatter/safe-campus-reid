import argparse


class ArgumentsBase(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        parser = self.parser
        parser.add_argument('-ddir', '--data-directory', default="static/dataset/safe-campus",
                            help='data directory of the image dataset', metavar='DIR')

        # gpu ids
        parser.add_argument('-gs', '--gpu-ids', type=int, nargs='+',
                            default=[0], help='ids of gpu devices to train the network')

        # the data loading setting
        parser.add_argument('-ds', '--dataset-name', type=str, default="market1501",
                            help='dataset name used for training')
        parser.add_argument('-b', '--batch-size', default=16,
                            type=int, help='mini-batch size')
        parser.add_argument('-nw', '--num-workers', default=0,
                            type=int, help='workers for loading data synchronously')

        # model info
        parser.add_argument('-model', '--model-name', type=str, default="external-bnneck-ibn-a",
                            help='model name which defines the structure of model backbones')

        # training/testing configuration version
        parser.add_argument('-vs', '--version', type=str, default='train-all_flips_duke-large-input',
                            help='defines model architecture, training settings and metrics')

    def parse_args(self):
        return self.parser.parse_args()
