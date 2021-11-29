from .cluster import construct_model
from .lagnet import Model as LAG
from config import args, config
from .vit.make_model import make_model

def get_cluster_model():
    model1 = construct_model(args['cluster']['train'], config['cluster']['train'])
    model2 = construct_model(args['cluster']['train2'], config['cluster']['train2'])
    model3 = construct_model(args['cluster']['test'], config['cluster']['test'])

    return model1, model2, model3

def get_lagnet_model():
    model = LAG(args['lagnet'])
    return model

def get_vit_model(loader):
    model = make_model(config['vit'],loader['vit']['num_classes'],loader['vit']['cam_num'],loader['vit']['view_num'])
    return model

def get_model(loader):
    model = {}

    model['cluster1'], model['cluster2'], model['test'] = get_cluster_model()
    model['lagnet'] = get_lagnet_model()
    model['vit'] = get_vit_model(loader)
    
    return model




