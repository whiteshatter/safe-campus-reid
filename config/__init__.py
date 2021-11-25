from .cluster_config.arguments_test import ArgumentsTest
from .cluster_config.arguments_train import ArgumentsTrainVal
from .cluster_config.config import get_config

from .lagnet_config.option import args as lagnet_arg

# 总配置
args = {}
config = {}

# cluster的相关配置
cluster_test_args = ArgumentsTest().parse_args()
cluster_train_args = ArgumentsTrainVal().parse_args()

args['cluster'] = {}
args['cluster']['train'] = cluster_train_args
args['cluster']['test'] = cluster_test_args

config['cluster'] = {}
config['cluster']['train'] = get_config(cluster_train_args)
config['cluster']['test'] = get_config(cluster_test_args)

# lagnet
args['lagnet'] = lagnet_arg
config['lagnet'] = lagnet_arg