"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.InsectDataLoader import InsectDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import pandas as pd
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset directory')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--classes', type=str, default=5, 
                        help='comma separated class names (e.g. bee,butterfly,...) or number [4,5,6] for default class list')
    parser.add_argument('--use_classes', type=str, default=None, 
                        help='Names of classes to load samples from. Comma separated class names (e.g. bee,butterfly,...) or a predefined list [6A, 6B, ...] for default class list')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--split_file', type=str, default=None, help='filename of train/test split file')
    return parser.parse_args()


def load_dataset(dataset_dir, args_classes):
    if args_classes=="4":
        classes = InsectDataLoader.CLASSES_4
    elif args_classes=="5":
        classes = InsectDataLoader.CLASSES_5
    elif args_classes=="6":
        classes = InsectDataLoader.CLASSES_6
    elif isinstance(args_classes, str):
        classes = args_classes.split(",")
    else:
        raise RuntimeError("Unsupported classes: " + str(args_classes))
    
    # dataset_dir = '../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-03_23-04-52'
    full_dataset = InsectDataLoader(root=dataset_dir, class_names=classes)

    # split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print("train, test size:", len(train_dataset), len(test_dataset))

    # data loaders
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    return classes, train_dataset, test_dataset, trainDataLoader, testDataLoader


def test(model, loader, classes, vote_num=1):
    num_classes = len(classes)
    classifier = model.eval()
    correct = 0
    total = 0
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    # Save preds for each sample
    # [[sample_path, target_id, target_name, cla0_pred, c1_pred, c2_pred, ...], ...]
    pred_per_sample = [] 

    for j, (points, target, path) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_classes).cuda()

        for _ in range(vote_num):
            # pred range [-inf, 0)
            pred, _, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        # Update the overall correct count and total count
        correct += pred_choice.eq(target.data).cpu().sum().item()
        total += target.size(0)

        # Update the correct count and total count per class
        for i in range(num_classes):
            correct_per_class[i] += (pred_choice[target == i] == i).sum().item()
            total_per_class[i] += (target == i).sum().item()

        # add each sample pred to list
        for pred1,choice1,target1,path1 in zip(pred.detach().cpu().numpy(), pred_choice.detach().cpu().numpy(), target.detach().cpu().numpy(), path):
            target_name = classes[target1]
            pred_per_sample.append( [path1, target_name, target1, choice1, *pred1] )

    overall_accuracy = correct / total
    accuracy_per_class = correct_per_class / total_per_class

    # accuracy_per_class is of type torch.Size([4]) torch.float32, ZB: tensor([1., 0., 0., 0.])
    return overall_accuracy, accuracy_per_class, pred_per_sample


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    classes, use_classes, _, _, _, test_data_loader = InsectDataLoader.load_dataset(
            args.dataset_dir, class_names=args.classes, use_classes=args.use_classes, batch_size=args.batch_size, \
            train_split=0.1, split_file=args.split_file)
    log_string("Ordered class names: " + str(classes))
    log_string("Using classes: " + str(use_classes))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(len(classes), normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_accs, pred_per_class = test(classifier.eval(), test_data_loader, vote_num=args.num_votes, classes=classes)
        class_accs = class_accs.detach().cpu().numpy()
        class_accs_fmt = ["%.3f" % acc for acc in class_accs]
        mean_class_acc = np.mean(class_accs, axis=0)

        log_string('Test Instance Accuracy: %f, Class Accuracy: %f, %s, pred len: %f' % (instance_acc, mean_class_acc, class_accs_fmt, len(pred_per_class)))
        
        # Save predictions
        fragments_df = pd.DataFrame(pred_per_class, \
                columns=["sample_path", "target_name", "target_id", "pred_choice", *classes])
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        fragments_df.to_csv(str(experiment_dir)+f"/logs/preds_per_sample_{timestr}.csv", index=False, header=True, decimal='.', sep=',', float_format='%.4f')


if __name__ == '__main__':
    args = parse_args()
    main(args)
