"""
Author: Benny, Marc
Date: Nov 2019

Run:
python train_classification.py --model pointnet2_cls_msg --classes 5 --batch_size 8 --epoch 20 --dataset_dir ../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-03_23-04-52


"""

import os
import sys
import time
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.InsectDataLoader import InsectDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset directory')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--classes', type=str, default="6B", 
                        help='Names of classes in order! Comma separated class names (e.g. bee,butterfly,...) or a predefined list [6A, 6B, ...] for default class list')
    parser.add_argument('--use_classes', type=str, default=None, 
                        help='Names of classes to load samples from. Comma separated class names (e.g. bee,butterfly,...) or a predefined list [6A, 6B, ...] for default class list')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--epoch', default=20, type=int, help='number of epoch in training')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--split_file', type=str, default=None, help='filename of train/test split file')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_classes=40):
    classifier = model.eval()
    correct = 0
    total = 0
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    for j, (points, target, path) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        # points is of shape (batch_size, num_points) and contains the data to classify
        points = points.transpose(2, 1)
        # pred is an array of shape (batch_size, num_classes) with a float value for each class
        pred, _, _ = classifier(points)
        # pred_choice is of shape (batch_size, 1) with an int for the index for the predicted class
        pred_choice = pred.data.max(1)[1]

        # Update the overall correct count and total count
        correct += pred_choice.eq(target.data).cpu().sum().item()
        total += target.size(0)

        # Update the correct count and total count per class
        for i in range(num_classes):
            correct_per_class[i] += (pred_choice[target == i] == i).sum().item()
            total_per_class[i] += (target == i).sum().item()

    overall_accuracy = correct / total
    accuracy_per_class = correct_per_class / total_per_class

    # accuracy_per_class is of type torch.Size([4]) torch.float32, ZB: tensor([1., 0., 0., 0.])
    return overall_accuracy, accuracy_per_class



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # DEBUG
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    '''CREATE DIRS'''
    exp_dir = Path('./log/classification')
    if args.log_dir is None:
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    classes, use_classes, train_dataset, test_dataset, train_data_loader, test_data_loader = InsectDataLoader.load_dataset(
        args.dataset_dir, class_names=args.classes, use_classes=args.use_classes, batch_size=args.batch_size, \
            train_split=0.1, split_file=args.split_file)
    log_string("Ordered class names: " + str(classes))
    log_string("Using classes: " + str(use_classes))
    print("train, test size:", len(train_dataset), len(test_dataset))

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(len(classes), normal_channel=False)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_class_accs_fmt = np.array([])

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target, path) in tqdm(enumerate(train_data_loader, 0), total=len(train_data_loader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat, activations = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f. Evaluating now...' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_accs = test(classifier.eval(), test_data_loader, num_classes=len(classes))
            class_accs = class_accs.detach().cpu().numpy()
            class_accs_fmt = ["%.3f" % acc for acc in class_accs]
            mean_class_acc = np.mean(class_accs, axis=0)
            
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
            if (mean_class_acc >= best_class_acc):
                best_class_acc = mean_class_acc
                best_class_accs_fmt = class_accs_fmt
            log_string('Test Instance Accuracy: %.3f, Class Accuracy: %.3f, %s' % (instance_acc, mean_class_acc, class_accs_fmt))
            log_string('Best Instance Accuracy: %.3f, Class Accuracy: %.3f, %s' % (best_instance_acc, best_class_acc, best_class_accs_fmt))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': mean_class_acc,
                    'class_accs': class_accs,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
