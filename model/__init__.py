from .cluster import construct_model
from .lagnet import Model as LAG
from config import args, config
from .vit.make_model import make_model

def get_cluster_model():
    model1 = construct_model(args['cluster']['train'], config['cluster']['train'])
    model2 = construct_model(args['cluster']['train2'], config['cluster']['train2'])
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
    return model1, model2

def get_lagnet_model():
    model = LAG(args['lagnet'])
    return model

def get_vit_model(loader):
    model = make_model(config['vit'],loader['vit']['num_classes'],loader['vit']['cam_num'],loader['vit']['view_num'])
    return model

def get_model(loader):
    model = {}

    model['cluster1'], model['cluster2'] = get_cluster_model()
    # model['lagnet'] = get_lagnet_model()
    # model['vit'] = get_vit_model(loader)
    
    return model




