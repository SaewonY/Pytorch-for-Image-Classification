import os
import gc
import math
import numpy as np
import pandas as pd
import time
import argparse

import torch
from torch import nn, cuda

from model import Baseline, Resnet18, Resnet50, Resnext50, Resnext101
import torchvision.models as models
from dataloader import make_loader, TestDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transforms import get_transform

from utils import count_parameters, seed_everything, AdamW, CosineAnnealingWithRestartsLR

import nsml
from nsml import DATASET_PATH


resnet50_lists = (
    ('team_26/airush1/1035', '8'), 
    ('team_26/airush1/1036', '9'),
    ('team_26/airush1/1042', '8'),
    ('team_26/airush1/1043', '8'),
    ('team_26/airush1/679', '9'), 
)


def model_infer(model, image_path, df_path):
    test_meta_data = pd.read_csv(df_path, delimiter=',', header=0)
    batch_size = 256
    num_classes = 350
    target_size = (224, 224)
    num_workers = 4
    device = args.device
    tta = 2

    transforms = get_transform(target_size, args.test_augments, args.augment_ratio, is_train=False)
    dataset = TestDataset(image_path, test_meta_data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    resnet_preds = []
    for i, (session, checkpoint) in enumerate(resnet50_lists):

        nsml.load(checkpoint=checkpoint, session=session)
        model.to(device)
        model.eval()

        tta_preds = []

        # 시간이 부족할 경우 tta 줄여보자
        # if i in [3, 4]:
        #     tta = 1

        for _ in range(tta):
            print("tta {} predict".format(_+1))
            preds = np.zeros((len(dataloader.dataset), num_classes))
            
            with torch.no_grad():
                for i, image in enumerate(dataloader):
                    image = image.to(device)
                    output = model(image) # output shape (batch_num, num_classes)
                    
                    preds[i*batch_size: (i+1)*batch_size] = output.detach().cpu().numpy()

            tta_preds.append(preds)
            del preds; gc.collect()

        tta_preds = np.mean(tta_preds, axis=0) # mean single model tta
        resnet_preds.append(tta_preds)
        del tta_preds; gc.collect()
    
    resnet_preds = np.mean(resnet_preds, axis=0) # mean between model weights
    del model, dataset, dataloader; gc.collect()

    return resnet_preds


def model1_infer(model1, image_path, df_path):
    test_meta_data = pd.read_csv(df_path, delimiter=',', header=0)

    batch_size = 64
    num_classes = 350
    target_size = (158, 158)
    num_workers = 4
    device = args.device
    tta = 2

    transforms = get_transform(target_size, args.test_augments, args.augment_ratio, is_train=False)
    dataset = TestDataset(image_path, test_meta_data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    dense_preds = []
    for i, (session, checkpoint) in enumerate(densenet_lists):

        nsml.load(checkpoint=checkpoint, session=session)
        model1.to(device)
        model1.eval()

        tta_preds = []

        for _ in range(tta):
            print("tta {} predict".format(_+1))
            preds = np.zeros((len(dataloader.dataset), num_classes))
            
            with torch.no_grad():
                for i, image in enumerate(dataloader):
                    image = image.to(device)
                    output = model(image) # output shape (batch_num, num_classes)
                    
                    preds[i*batch_size: (i+1)*batch_size] = output.detach().cpu().numpy()

            tta_preds.append(preds)
            del preds; gc.collect()

        tta_preds = np.mean(tta_preds, axis=0) # mean single model tta
        dense_preds.append(tta_preds)
        del tta_preds; gc.collect()
    
    dense_preds = np.mean(dense_preds, axis=0) # mean between model weights
    
    return dense_preds


def bind_model(model, model1):
    def save(dir_name, *args, **kwargs):
        state = {
            'model': model.state_dict(),
        }
        state1 = {
            'model': model1.state_dict(),
        }
        torch.save(state, os.path.join(dir_name, 'state_dict.pkl'))
        torch.save(state1, os.path.join(dir_name, 'state_dict1.pkl'))
        print("models saved!")

    def load(dir_name, *args, **kwargs):
        
        try:
            state = torch.load(os.path.join(dir_name, 'state_dict.pkl'))
            a = model.load_state_dict(state['model'])
            print("resnet50 loaded")
        
        except:
            state1 = torch.load(os.path.join(dir_name, 'state_dict1.pkl'))
            b = model1.load_state_dict(state1['model'])
            print("densenet loaded" )        

    def infer(test_image_data_path, test_meta_data_path):
        resnet_preds = model_infer(model, test_image_data_path, test_meta_data_path) # resnet50 - 5fold * tta
        densenet_preds = model1_infer(model1, test_image_data_path, test_meta_data_path) # densenet
        predict_vector  = 0.9*resnet_preds + 0.1*densenet_preds

        # predict_vector = ((resnet_preds + densenet_preds) / 2 + resnet_preds) / 2
        predict_vector = np.argmax(predict_vector, axis=1)
        return predict_vector  

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    # reserved for nsml
    arg("--mode", type=str, default="train")
    arg("--iteration", type=str, default='0')
    arg("--pause", type=int, default=0)

    arg('--model', type=str, default='Resnet50')
    # arg('--input_size', type=int, default=224)
    arg('--test_augments', default='resize, horizontal_flip', type=str)
    arg('--augment_ratio', default=0.5, type=float, help='probability of implementing transforms')
    arg('--device', type=int, default=0)
    arg('--hidden_size', type=int, default=128)
    args = parser.parse_args()

    device = args.device

    SEED = 2019
    seed_everything(SEED)

    model = Resnet50(350, dropout=False)
    model = model.to(device)

    # model1 = Baseline(args.hidden_size, 350)
    # model1 = models.densenet201(pretrained=True)
    # model1.classifier = nn.Linear(1920, 350)

    model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
    model.classifier = nn.Sequential(nn.Dropout(0.2),
                        nn.Linear(1280, args.num_classes))


    # model1 = Resnext101(350, dropout=False)

    model1 = model1.to(device)

    bind_model(model, model1)
    if args.pause:
        nsml.paused(scope=locals())

    nsml.save('final')