'''
@author: Xu Yan, Marc Böge
'''
import os
import numpy as np
import warnings
import pickle
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

class InsectDataLoader(Dataset):
    # A uses old order; B uses new order
    CLASSES_4A = ["bee","butterfly","dragonfly","wasp"]
    CLASSES_5A = ["bee","butterfly","dragonfly","wasp","insect"]
    CLASSES_6A = ["bee","butterfly","dragonfly","wasp","insect","other"]
    CLASSES_6B = ["other","insect","bee","butterfly","dragonfly","wasp"]
    CLASSES_7A = ["bee","butterfly","dragonfly","wasp","other","insect","bumblebee"]
    CLASSES_7B = ["other","insect","bee","butterfly","dragonfly","wasp","bumblebee"]

    def __init__(self, root, classes=CLASSES_6B, use_classes=None):
        """
        Args:
            root (str): root path of dataset directory
            classes (_type_, optional): Class list ordered by id, beginning at 0. Defaults to CLASSES_6B.
            use_classes (_type_, optional): Load samples of only these classes. Defaults to None = all classes.
        """
        self.root = root
        # <class_name>:<class_id>
        self.classes = dict(zip(classes, range(len(classes))))

        self.samples = []
        for f in Path(root).glob("*/*.csv"):
            clas = f.parent.name
            if use_classes is not None and clas not in use_classes:
                # skip this sample if class is not used
                continue
            point_set = np.loadtxt(f, delimiter=',', skiprows=1, usecols=(0,1,2)).astype(np.float32)
            # point_set = np.zeros((10,10))
            rel_path = str(Path(clas) / f.name)
            self.samples.append( (point_set, clas, rel_path) )


    def __len__(self):
        return len(self.samples)

    def _get_item(self, index):
        sample = self.samples[index]
        point_set = sample[0]
        class_id = self.classes[sample[1]]
        rel_path = sample[2]
        return point_set, class_id, rel_path

    def __getitem__(self, index):
        return self._get_item(index)

    @staticmethod
    def load_dataset(dataset_dir, args_classes, batch_size=8, train_split=0.8):
        if args_classes=="4A":
            classes = InsectDataLoader.CLASSES_4A
        elif args_classes=="5A":
            classes = InsectDataLoader.CLASSES_5A
        elif args_classes=="6A":
            classes = InsectDataLoader.CLASSES_6A
        elif args_classes=="6B":
            classes = InsectDataLoader.CLASSES_6B
        elif args_classes=="7A":
            classes = InsectDataLoader.CLASSES_7A
        elif args_classes=="7B":
            classes = InsectDataLoader.CLASSES_7B
        elif isinstance(args_classes, str):
            classes = args_classes.lower().split(",")
        else:
            raise RuntimeError("Unsupported classes: " + str(args_classes))
        
        # dataset_dir = '../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-03_23-04-52'
        full_dataset = InsectDataLoader(root=dataset_dir, classes=classes)

        if train_split <= 0.0:
            # put all in test
            test_data_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
            print("train, test size:", 0, len(full_dataset))
            return classes, None, None, full_dataset, test_data_loader
        
        # else:
        # split in train and test
        train_size = int(train_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        print("train, test size:", len(train_dataset), len(test_dataset))

        # data loaders
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        return classes, train_dataset, test_dataset, train_data_loader, test_data_loader


if __name__ == '__main__':
    import torch
    # data = InsectDataLoader('../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-03_23-04-52')
    data = InsectDataLoader('../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-21_23-38-04')
    print("samples:", data.__len__())
    DataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False, num_workers=1)
    pc, clas, file = next(iter(DataLoader))
    print(pc.shape, pc.dtype)
    print(clas.shape, clas.dtype)
    print(file)
    # result:
    # torch.Size([4, 10, 10]) torch.float64
    # torch.Size([4]) torch.int64
    # ('wasp\\wasp_h9_2_69.csv', 'bee\\bee_m3_0_201.csv', 'insect\\insect_h7_25_1.csv', 'butterfly\\butterfly_h7_32_36.csv')

    for cl,pc1,pth in zip(clas,pc,file):
        print(cl.cpu(), cl.shape, pc1.shape, pth)
