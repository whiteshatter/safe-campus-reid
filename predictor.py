# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import atexit
import bisect
from collections import deque

import cv2
import torch
import torch.multiprocessing as mp
from PIL import Image
from trainer import Trainer
import data
from model import get_model
from config import args, config

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


class DefaultPredictor:
    def __init__(self):
        model = get_model()

    def __call__(self, image):
        image = self.img_transform(image).unsqueeze(0)
        inputs = image
        features = torch.FloatTensor()
        ff = torch.FloatTensor(inputs.size(0), 3584).zero_() 
        for i in range(2):
            if i==1:
                inputs = fliphor(inputs)
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff)) 
        features = torch.cat((features, ff), 0)
        return features.numpy()



class FeatureExtractionDemo(object):
    def __init__(self):
        self.predictor = DefaultPredictor()

    def run_on_image(self, original_image_path):
        riginal_image = Image.open(original_image_path).convert('RGB')
        predictions = self.predictor(riginal_image)
        return predictions

    # def run_on_loader(self, data_loader):
    #     if self.parallel:
    #         buffer_size = self.predictor.default_buffer_size

    #         batch_data = deque()

    #         for cnt, batch in enumerate(data_loader):
    #             batch_data.append(batch)
    #             self.predictor.put(batch["images"])

    #             if cnt >= buffer_size:
    #                 batch = batch_data.popleft()
    #                 predictions = self.predictor.get()
    #                 yield predictions, batch["targets"].cpu().numpy(), batch["camids"].cpu().numpy()

    #         while len(batch_data):
    #             batch = batch_data.popleft()
    #             predictions = self.predictor.get()
    #             yield predictions, batch["targets"].cpu().numpy(), batch["camids"].cpu().numpy()
    #     else:
    #         for batch in data_loader:
    #             predictions = self.predictor(batch["images"])
    #             yield predictions, batch["targets"].cpu().numpy(), batch["camids"].cpu().numpy()