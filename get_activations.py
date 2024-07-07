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
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
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
    full_dataset = InsectDataLoader(root=dataset_dir, classes=classes)

    # split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print("train, test size:", len(train_dataset), len(test_dataset))

    # data loaders
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    return classes, train_dataset, test_dataset, trainDataLoader, testDataLoader


def get_activations(classifier, loader, classes):
    classifier = classifier.eval()
    activations_per_sample = [] # [[sample_path, target_name [c0_activations, c1_act., c2_act., ...], ...]

    for j, (points, target, path) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)

        # also get activations from fc2-layer; see pointnet2_cls_msg class
        prediciton, _, fc2_activations = classifier(points)

        # for each sample add activations to a list
        for activations1, target1, path1 in zip(fc2_activations.detach().cpu().numpy(), target.detach().cpu().numpy(), path):
            target_name = classes[target1]
            activations_per_sample.append( [path1, target_name, *activations1] )

    return activations_per_sample


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

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
    classes, _, _, _, test_data_loader = load_dataset(args.dataset_dir, args.classes)
    log_string("Using classes: " + str(classes))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(len(classes), normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        activations_per_sample = get_activations(classifier, test_data_loader, classes)
        activations_header = ["act_"+str(i) for i in range(len(activations_per_sample[0])-2)]
        print("activations:", len(activations_header))

        # Save 
        fragments_df = pd.DataFrame(activations_per_sample, \
                columns=["sample_path", "target_name", *activations_header])
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        fragments_df.to_csv(str(experiment_dir)+f"/logs/activations_per_class_{timestr}.csv", index=False, header=True, decimal='.', sep=',', float_format='%.4f')


if __name__ == '__main__':
    args = parse_args()
    main(args)
