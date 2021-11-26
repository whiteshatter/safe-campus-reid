from .cluster_config.arguments_test import ArgumentsTest
from .cluster_config.arguments_first_train import ArgumentsTrainVal1
from .cluster_config.arguments_second_train import ArgumentsTrainVal2
from .cluster_config.config import get_config

from .lagnet_config.option import args as lagnet_arg

from config.vit_config.defaults import _C as cfg

# 总配置
args = {}
config = {}

# cluster的相关配置
cluster_test_args = ArgumentsTest().parse_args()
cluster_first_train_args = ArgumentsTrainVal1().parse_args()
cluster_second_train_args = ArgumentsTrainVal2().parse_args()

args['cluster'] = {}
args['cluster']['train'] = cluster_first_train_args
args['cluster']['train2'] = cluster_second_train_args
args['cluster']['test'] = cluster_test_args

config['cluster'] = {}
config['cluster']['train'] = get_config(cluster_first_train_args)
config['cluster']['train2'] = get_config(cluster_second_train_args)
config['cluster']['test'] = get_config(cluster_test_args)

# lagnet
args['lagnet'] = lagnet_arg
config['lagnet'] = lagnet_arg

# vit
config['vit'] = cfg