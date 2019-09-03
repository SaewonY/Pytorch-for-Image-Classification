import gc
import os
import time
import numpy as np
import argparse
import pathlib
import tqdm
from collections import defaultdict, Counter

from model import Baseline, Resnet18, Resnet50, Resnext50, Resnext101
import nsml
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, cuda
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable 
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from dataloader import make_loader, TestDataset
from transforms import get_transform
from utils import count_parameters, seed_everything, AdamW, CosineAnnealingWithRestartsLR
from customs import mixup_data, mixup_criterion
from nsml import DATASET_PATH
from efficientnet_pytorch import EfficientNet


def to_numpy(t):
    return t.cpu().detach().numpy()

def bind_model(model_nsml):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict1.pkl')
        state = {
                    'model': model_nsml.state_dict(),
                }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict1.pkl')
        state = torch.load(save_state_path)
        model_nsml.load_state_dict(state['model'])
        
    def infer(test_image_data_path, test_meta_data_path):
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)
        
        tta = args.tta
        num_classes = args.num_classes
        target_size = (args.input_size, args.input_size)
        batch_size = 200 
        num_workers = 4
        device = 0

        transforms = get_transform(target_size, args.test_augments, args.augment_ratio, is_train=False)
        dataset = TestDataset(test_image_data_path, test_meta_data, transform=transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model_nsml.to(device)
        model_nsml.eval()
        total_predict = []
        
        for _ in range(tta):
            print("tta {} predict".format(_+1))
            prediction = np.zeros((len(dataloader.dataset), num_classes))

            with torch.no_grad():
                for i, image in enumerate(dataloader):
                    image = image.to(device)
                    output = model_nsml(image) # output shape (batch_num, num_classes)
                    
                    prediction[i*batch_size: (i+1)*batch_size] = output.detach().cpu().numpy()

                    total_predict.append(prediction)
                    del prediction; gc.collect()

        total_predict = np.mean(total_predict, axis=0) # mean tta predictions
        predict_vector = np.argmax(total_predict, axis=1) # return index shape of (138343)

        return predict_vector # this return type should be a numpy array which has shape of (138343)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def make_folds(df, n_folds: int) -> pd.DataFrame:
    
    cls_counts = Counter([classes for classes in df['tag']])
    fold_cls_counts = defaultdict()
    for class_index in cls_counts.keys():
        fold_cls_counts[class_index] = np.zeros(n_folds, dtype=np.int)

    df['fold'] = -1
    pbar = tqdm.tqdm(total=len(df))

    def get_fold(row):
        class_index = row['tag']
        counts = fold_cls_counts[class_index]
        fold = np.argmin(counts)
        counts[fold] += 1
        fold_cls_counts[class_index] = counts
        row['fold']=fold
        pbar.update()
        return row
    
    df = df.apply(get_fold, axis=1)
    return df


def make_validation(meta_df, label_path, NROWS=None, DEBUG=False):

    if DEBUG:
        label_matrix = np.load(label_path)[:NROWS]
    else:
        label_matrix = np.load(label_path)

    
    b = []
    for i in range(label_matrix.shape[0]):
        tag = np.argmax(label_matrix[i])
        b.append(tag)

    tag_df = pd.DataFrame(b, columns=['tag'])
    df = pd.concat([meta_df, tag_df], axis=1)
    folds = make_folds(df, n_folds=3)
    
    return folds

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    # DONOTCHANGE: They are reserved for nsml
    arg('--mode', type=str, default='train')
    arg('--iteration', type=str, default='0')
    arg('--pause', type=int, default=0)
    
    # custom args
    arg('--epochs', type=int, default=30)
    arg('--input_size', type=int, default=158)
    arg('--batch_size', type=int, default=512)
    arg('--num_workers', type=int, default=4)
    arg('--gpu_num', type=int, nargs='+', default=[0])
    arg('--model', type=str, default='resnet50')
    arg('--optimizer', type=str, default='Adam') # SGD
    arg('--scheduler', type=str, default='StepLR', help='scheduler in steplr, plateau, cosine')
    arg('--hidden_size', type=int, default=128)
    arg('--num_classes', type=int, default=350, help='num tags to predict') # Fixed
    arg('--train_augments', default='resize, horizontal_flip', type=str)
    arg('--test_augments', default='resize, horizontal_flip', type=str)
    arg('--augment_ratio', default=0.5, type=float, help='probability of implementing transforms')
    arg('--mixup_loss', default=False, type=bool)
    arg('--lr', type=float, default=2.5e-4)
    arg('--fold_num', type=int, default=0, help='select fold number to train on')
    arg('--tta', type=int, default=1, help='test time augmentation')
    arg('--log_interval', type=int, default=300, help='for log checking')
    arg('--device', type=int, default=0)
    arg('--DEBUG', default=False, type=bool, help='if true debugging mode')
    args = parser.parse_args()

    # commands examples
    # for debugging => nsml run -d airush1 -a "--DEBUG True --num_workers 0" (with gpu)
    # for debugging => nsml run -d airush1 -g 0 -a "--DEBUG True --num_workers 0" (with cpu)
    # for training => nsml run -d airush1 -a "--model Resnet50 --batch_size 256--train_augments 'resize, horizontal_flip, random_rotate'" -m "saewon"


    # fix seed for reproduction
    SEED = 2019
    seed_everything(SEED)

    device = args.device
    use_gpu = cuda.is_available()

    if use_gpu:
        print("enable gpu use")
    else:
        print("enable cpu for debugging")


    target_size = (args.input_size, args.input_size)

    if args.model == 'base':
        assert args.input_size == 128
        model = Baseline(args.hidden_size, args.num_classes)
    elif args.model == 'resnet18': 
        model = Resnet18(args.num_classes, dropout=False)
    elif args.model == 'resnet50':
        model = Resnet50(args.num_classes, dropout=False)
    elif args.model == 'efficient':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, args.num_classes)
    elif args.model == 'densenet201':
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(1920, args.num_classes)
    elif args.model == 'resnext50':
        model = Resnext50(args.num_classes, dropout=False)
    elif args.model == 'resnext101':
        model = Resnext101(args.num_classes, dropout=False)
    elif args.model == 'mobilenet':
        assert args.input_size == 158
        model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
        model.classifier = nn.Sequential(nn.Dropout(0.2),
                            nn.Linear(1280, args.num_classes))
    else:
        raise NotImplementedError

    if use_gpu:
        model = model.to(device)
    
    if args.optimizer == 'Adam':
        # optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=0.00025)
        optimizer = AdamW(model.parameters(), args.lr, weight_decay=0.000025)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.025)
    else:
        raise NotImplementedError

    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    elif args.scheduler == 'cosine':
        eta_min = 1e-5
        T_max = 10
        T_mult = 1
        restart_decay = 0.97
        scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)
    elif args.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        raise NotImplementedError


    criterion = nn.CrossEntropyLoss() 

    num_parameters = count_parameters(model)
    print("number of parameters that can be trained: {}".format(num_parameters))
    print()

    # DONOTCHANGE
    bind_model(model)
    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == "train":

        # Warning: Do not load data before this line
        image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
        meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')

        print("start making folds df")

        if args.DEBUG:
            NROWS = 1000
            meta_df = pd.read_csv(meta_path, delimiter=',', header=0, nrows=NROWS)
            df = make_validation(meta_df, label_path, NROWS=NROWS, DEBUG=args.DEBUG)
            train_df = df
            valid_df = df
        else:
            NROWS = None
            meta_df = pd.read_csv(meta_path, delimiter=',', header=0, nrows=NROWS)
            df = make_validation(meta_df, label_path)
        
            train_df = df.loc[df['fold'] != args.fold_num].reset_index(drop=True)
            valid_df = df.loc[df['fold'] == args.fold_num].reset_index(drop=True)

        train_transforms = get_transform(target_size, args.train_augments, args.augment_ratio)
        valid_transforms = get_transform(target_size, args.test_augments, args.augment_ratio, is_train=False)
        train_loader = make_loader(train_df, image_dir, train_transforms, args.batch_size, args.num_workers)
        valid_loader = make_loader(valid_df, image_dir, valid_transforms, args.batch_size, args.num_workers)
        print("train batches: {}".format(len(train_loader)))
        print("valid batches: {}".format(len(valid_loader)))
        print("batch_size: {}".format(args.batch_size))
        print()

        best_val_acc = 0
        grad_clip_step = 100
        grad_clip = 100
        step = 0
        accumulation_step = 2

        for epoch_idx in range(1, args.epochs + 1):

            start_time = time.time()

            train_loss = 0
            train_total_correct = 0
            model.train()
            optimizer.zero_grad()

            for batch_idx, (image, tags) in enumerate(train_loader):
                if use_gpu:
                    image = image.to(device)
                    tags = tags.to(device)

                if args.mixup_loss:
                    inputs, targets_a, targets_b, lam = mixup_data(image, tags, alpha=0.4, device=device)
                    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                    output = model(inputs)
                    loss = mixup_criterion(criterion, output.to(device), targets_a.to(device), targets_b.to(device), lam)
                else:
                    output = model(image)
                    loss = criterion(output, tags)

                # gradient explosion prevention
                if step > grad_clip_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                step += 1

                loss.backward()
                
                if (batch_idx+1) % accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                output_prob = F.softmax(output, dim=1)
                predict_vector = np.argmax(to_numpy(output_prob), axis=1)
                label_vector = to_numpy(tags)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)

                if batch_idx % args.log_interval == 0:
                    print('Batch {} / {}: Batch Loss {:.4f} / Batch Acc {:.2f}'.format(
                        batch_idx, len(train_loader), loss.item(), accuracy))
                train_loss += loss.item() / len(train_loader)
                train_total_correct += bool_vector.sum()

            model.eval()
            valid_loss = 0
            valid_total_correct = 0

            with torch.no_grad():
                for batch_idx, (image, tags) in enumerate(valid_loader):
                    if use_gpu:
                        image = image.to(device)
                        tags = tags.to(device)

                    output = model(image)
                    loss = criterion(output, tags)

                    output_prob = F.softmax(output, dim=1)
                    predict_vector = np.argmax(to_numpy(output_prob), axis=1)
                    label_vector = to_numpy(tags)
                    bool_vector = predict_vector == label_vector

                    valid_loss += loss.item() / len(valid_loader)
                    valid_total_correct += bool_vector.sum()
    
            elapsed = time.time() - start_time

            train_acc = train_total_correct/len(train_loader.dataset)
            val_acc = valid_total_correct/len(valid_loader.dataset)

            # best val_acc checkpoint
            if val_acc > best_val_acc:
                print("val_acc has improved")
                best_val_acc = val_acc
                nsml.save('best_acc')
            else:
                print("val_acc has not improved")

            lr = [_['lr'] for _ in optimizer.param_groups]

            if args.scheduler == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()

            nsml.save(epoch_idx)
            
            print("Epoch {}/{}  train_loss: {:.5f}  valid_loss {:.5f}  train_acc: {:.3f}  valid_acc: {:.3f}  lr: {:.6f}  elapsed: {:.0f}".format(
                   epoch_idx, args.epochs, train_loss, valid_loss, train_acc, val_acc, lr[0], elapsed))

            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                "train_loss": round(train_loss, 5),
                "valid_loss": round(valid_loss, 5),
                "train_acc": round(train_acc, 5),
                "valid_acc": round(val_acc, 5)
                })
            print()