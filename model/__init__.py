from .cluster import construct_model
from config import args, config


def get_cluster_model():
    return construct_model(args['cluster']['test'], config['cluster']['test'])


def get_model():
    model = {}

    model['cluster'] = get_cluster_model()
    return model




