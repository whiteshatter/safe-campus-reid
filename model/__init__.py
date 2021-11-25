from .cluster import construct_model
from .lagnet import Model as LAG
from config import args, config
import torch

def get_cluster_model():
    cluster_args = args['cluster']['test']
    model = construct_model(cluster_args, config['cluster']['test'])
    # save_path = cluster_args.restore_file
    # if save_path is not None or save_path != "":
    #     if cluster_args.gpu_ids is None:
    #         checkpoint = torch.load(save_path)
    #     else:
    #         loc = 'cuda:{}'.format(cluster_args.gpu_ids[0])
    #         checkpoint = torch.load(save_path, map_location=loc)
    #     print('==> Resume checkpoint {}'.format(save_path))
    #     if 'state_dict' in checkpoint:
    #         checkpoint = checkpoint['state_dict']
    #     if 'transfer' in cluster_args.version:
    #         checkpoint = {key: val for key, val in checkpoint.items() if 'classifier' not in key}
    #         msg = model.load_state_dict(checkpoint, strict=False)
    #         print(msg)
    #     else:
    #         model.load_state_dict(checkpoint)
    return model

def get_lagnet_model():
    model = LAG(args['lagnet'])
    return model

def get_model():
    model = {}

    model['cluster'] = get_cluster_model()
    model['lagnet'] = get_lagnet_model()
    
    return model




