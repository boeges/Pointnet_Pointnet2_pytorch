"""
Author: Benny
Date: Nov 2019

useage:
CUDA_VISIBLE_DEVICES=1 python test_classification.py --log_dir log/classification/msg_cls4A_e40_bs8_pts4096_split40shot_4 --model pointnet2_cls_msg --classes 4A --batch_size 8 --dataset_dir ../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_4/ --split_file train_test_split_40shot.txt



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

def test(model, loader, classes, vote_num=1):
    # evaluate on by original method
    overall_accuracy, accuracy_per_class, pred_per_sample = test1(model, loader, classes, vote_num)
    # evaluate per instance
    test_per_instance(pred_per_sample, classes)
    return overall_accuracy, accuracy_per_class, pred_per_sample


def test1(model, loader, classes, vote_num=1):
    num_classes = len(classes)
    classifier = model.eval()
    correct = 0
    total = 0
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    # Save preds for each sample
    # [[sample_path, target_id, target_name, cla0_pred, c1_pred, c2_pred, ...], ...]
    pred_per_class = [] 

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
            pred_per_class.append( [path1, target_name, target1, choice1, *pred1] )

    overall_accuracy = correct / total
    accuracy_per_class = correct_per_class / total_per_class

    # accuracy_per_class is of type torch.Size([4]) torch.float32, ZB: tensor([1., 0., 0., 0.])
    return overall_accuracy, accuracy_per_class, pred_per_class


def test_per_instance(pred_per_class, classes):
    def f2(v:str):
        vs = v.split("/")[-1].split(".")[-2].split("_")[-3:]
        return vs
    
    df = pd.DataFrame(pred_per_class, \
            columns=["sample_path", "target_name", "target_id", "pred_choice", *classes])

    df['scene'], df['instance'], df['frag'] = zip(*df['sample_path'].map(f2))

    aggs = {
        "target_name":"first",
        "target_id":"first",
        "pred_choice":"nunique",
        # "pred_choice":pd.Series.mode,
    }
    aggs_cl = {k:"mean" for k in classes}
    aggs.update(aggs_cl)
 
    df1 = df.groupby(["scene","instance"]).agg(aggs).rename(columns={'pred_choice': 'nunique'})
    df1[df1["target_name"]=="dragonfly"]

    df2 = df1[["bee", "butterfly", "dragonfly", "wasp"]].idxmax(axis=1).rename("pred_name")
    df1 = pd.concat([df1,df2], axis=1)

    # classes = ["bee", "butterfly", "dragonfly", "wasp"]
    cm = {v:k for k,v in enumerate(classes)}
    df1["pred_id"] = df1["pred_name"].map(cm)

    instance_count = df1.index.__len__()
    num_correct = (df1["target_name"] == df1["pred_name"]).sum()
    acc = num_correct/instance_count
    print(f"Overall Grouped-By-Instance Accuracy {acc:.3f}; Instances={instance_count}; Correct={num_correct}")
        
    unq = df1["target_name"].unique()
    accs = []
    for clas in unq:
        dfc = df1[df1["target_name"]==clas]
        instance_count = dfc.index.__len__()
        num_correct = (dfc["target_name"] == dfc["pred_name"]).sum()
        acc = num_correct/instance_count
        accs.append(acc)
        print(f"{clas:<10}: Grouped-By-Instance Accuracy {acc:.3f}; Instances={instance_count}; Correct={num_correct}")

    print("Unweighted mean Grouped-By-Instance Class Accuracy", np.mean(accs))


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''SET DIR'''
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
    # model_name = os.listdir(experiment_dir + '/')[0].split('.')[0]
    model_name = args.model
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
        path = str(experiment_dir)+f"/logs/pred_per_sample_{timestr}.csv"
        fragments_df.to_csv(path, index=False, header=True, decimal='.', sep=',', float_format='%.4f')
        log_string('Saved predictions csv as %s' % (path))



if __name__ == '__main__':
    args = parse_args()
    main(args)
