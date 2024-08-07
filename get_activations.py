"""
Author: Benny
Date: Nov 2019

run with:
python .\get_activations.py --model pointnet2_cls_msg --classes 6B --use_classes 6B --batch_size 8 --dataset_dir ..\..\datasets\insect\100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-23_12-17-56\ --log_dir 2024-07-25_22-10


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
    parser.add_argument('--model_class_num', type=int, default=None, help='number of classes for the model output')
    parser.add_argument('--classes', type=str, default="6B", 
                        help='Names of classes in order! Comma separated class names (e.g. bee,butterfly,...) or a predefined list [6A, 6B, ...] for default class list')
    parser.add_argument('--use_classes', type=str, default=None, 
                        help='Names of classes to load samples from. Comma separated class names (e.g. bee,butterfly,...) or a predefined list [6A, 6B, ...] for default class list')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def load_dataset(dataset_dir, args_classes, train_split=0.8):
    return 


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
    classes, use_classes, _, _, _, test_data_loader = InsectDataLoader.load_dataset(dataset_dir=args.dataset_dir, 
            class_names=args.classes, use_classes=args.use_classes, batch_size=args.batch_size, train_split=0.0)
    log_string("Ordered class names: " + str(classes))
    log_string("Using classes: " + str(use_classes))

    '''MODEL LOADING'''
    # model_path = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model_path = args.model
    model = importlib.import_module(model_path)

    model_class_num = args.model_class_num if args.model_class_num is not None else len(classes)
    classifier = model.get_model(model_class_num, normal_channel=args.use_normals)
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
        activations_path = str(experiment_dir)+f"/logs/activations_per_sample_{timestr}.csv"
        fragments_df.to_csv(activations_path, index=False, header=True, decimal='.', sep=',', float_format='%.4f')
        print("Saved to:", activations_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
