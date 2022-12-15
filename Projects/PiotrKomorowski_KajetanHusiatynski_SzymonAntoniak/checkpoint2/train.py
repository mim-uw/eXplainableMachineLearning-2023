#!/usr/bin/env python3
import argparse
import logging
import logging.handlers
import os
import pdb
import random
import re
import shutil
import multiprocessing
from PIL import Image

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader

from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP


def accuracy(output, target):#, topk=(1,)):
    with torch.no_grad():
        # print('TARGET: ', target, target.shape)
        # maxk = max(topk)
        batch_size = target.shape[0]
        # print('OUTPUT: ', output.shape, output)
        # _, pred = output.topk(k=1, dim=1)
        pred = output.argmax(dim=1)
        # pred = output.squeeze() > 0
        # print('PRED: ', pred.shape, pred)
        # print('TARGET: ', target.shape, target)
        # pred = pred.t()
        correct = pred.eq(target)
        # print(correct)
        acc = correct.float().sum().mul(100.0 / batch_size)
        return acc


def preprocess(args):

    print('prepare model')
    model = vit_LRP(pretrained=args.pretrained)
    model.head = nn.Linear(model.head.in_features, 3) # change number of outputs
    return model.to(args.device)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(args, model):
    print('training data loading')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_dataset = datasets.ImageFolder(
    args.train_dir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    print(train_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=True,
                             drop_last=False)

    # raise SystemExit(0)
    print('training data loading completed')

    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # clear file
    with open(args.temp_output, 'r+') as f:
        f.truncate(0)
    
    best_val_acc = torch.tensor(0) # as tensor because validation acc will be a tensor
    
    for epoch in range(args.num_epochs):
        model.train()
        print(f'epoch {epoch}/{args.num_epochs-1}')
        train_acc = 0
        train_count = 0

        for i, (image, target) in enumerate(train_loader):
            # if i > 0: break
            
            image, target = image.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(image)
            
            # print('TARGET: ', target.shape, target)
            # print('OUTPUT: ', output.shape, output)
            loss = loss_func(output.squeeze(), target)
            acc = accuracy(output, target)
            train_count += image.shape[0]
            train_acc += acc * image.shape[0]
            train_acc_avg = train_acc / train_count
            
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i != 0:
                with open(args.temp_output, 'a') as f:
                    f.write((f"Train, step #{i}/{len(train_loader)}, "
                            f"accuracy {train_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))
                print((f"Train, step #{i}/{len(train_loader)}, "
                            f"accuracy {train_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))

        
        val_acc = validate(args, model)
        
        is_best = val_acc > best_val_acc
        log_val = f'Val acc improved from {round(best_val_acc.item(), 3)} to {round(val_acc.item(), 3)}' if is_best \
            else f'Val acc has not improved form {round(best_val_acc.item(), 3)}'
        best_val_acc = max(val_acc, best_val_acc)
        print(log_val)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


    log = '\n'.join([
        f'raining results',
        f'- Accuracy: {train_acc_avg:.4f}'])

    print('train finish')
    print(log)


def validate(args, model):
    val_dataset = datasets.ImageFolder(
        args.val_dir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))
    
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                             drop_last=False)

    model.eval()

    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.BCEWithLogitsLoss()

    val_acc = 0
    val_count = 0

    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            image, target = image.to(args.device), target.to(args.device)

            output = model(image)
            
            loss = loss_func(output.squeeze(), target)
            acc = accuracy(output, target)
            val_count += image.shape[0]
            val_acc += acc * image.shape[0]
            val_acc_avg = val_acc / val_count


            if i % 100 == 0 and i != 0:
                with open(args.temp_output, 'a') as f:
                    f.write((f"Val, step #{i}/{len(val_loader)}, "
                            f"accuracy {val_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))
                print((f"Val, step #{i}/{len(val_loader)}, "
                            f"accuracy {val_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))




    log = '\n'.join([
        f'val results',
        f'- Accuracy: {val_acc_avg:.4f}'])

    # logging.info('train finish')
    print('val finish')
    # logging.info(log)
    print(log)

    return val_acc_avg

def test(args, model):

    test_dataset = datasets.ImageFolder(
        args.test_dir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))
    
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                             drop_last=False)

    

    model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.BCEWithLogitsLoss()

    print('testing')

    test_acc = 0
    test_count = 0
    with torch.no_grad():
        for i, (image, target) in enumerate(test_loader):
            # if i > 0: break
            image, target = image.to(args.device), target.to(args.device)

            # print('IMAGE: ', image.shape)
            # print('TARGET: ', target, target.shape)

            output = model(image)
            loss = loss_func(output.squeeze(), target)
            acc = accuracy(output, target)
            test_count += image.shape[0]
            test_acc += acc * image.shape[0]
            test_acc_avg = test_acc / test_count
            

            if i % 100 == 0:
                with open(args.temp_output, 'a') as f:
                    f.write((f"test, step #{i}/{len(test_loader)}, "
                            f"accuracy {test_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))
                print((f"Test, step #{i}/{len(test_loader)}, "
                            f"accuracy {test_acc_avg:.3f}, "
                            f"loss {loss:.3f}, "))

    log = '\n'.join([
        f'# Test Result',
        f'- acc: {test_acc_avg:.4f}'])


    # logging.info('test finish')
    print('test finish')
    # logging.info(log)
    print(log)




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=None,
                        help='directory path of train dataset')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='directory path of val dataset')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='directory path of test dataset')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='batch size of validation')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='batch size of inference')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--heatmap_scale', type=int, default=5000,
                        help='scale value for heatmap visualization')
    parser.add_argument('--pretrained', type=int, default=1,
                        choices=[0, 1],
                        help='whether model should be pretrained on imagenet')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--temp_output', type=str, default='output.txt',
                        help='file for temporal output')
                        

    args = parser.parse_args()


    model = preprocess(args)
    train(args, model)
    test(args, model)
    
