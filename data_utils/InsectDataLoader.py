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

# make key (scene_id, instance_id, frag_index).
# example: "dragonfly/dragonfly_h3_6_5.csv" becomes ("hn-dra-1", 6, 5).
def frag_filename_to_id(fn):
    return "_".join(fn.replace(".csv","").split("_")[-3:])

class InsectDataLoader(Dataset):
    # A uses old order; B uses new order
    CLASSES_4A = ["bee","butterfly","dragonfly","wasp"]
    CLASSES_5A = ["bee","butterfly","dragonfly","wasp","insect"]
    CLASSES_6A = ["bee","butterfly","dragonfly","wasp","insect","other"]
    CLASSES_6B = ["other","insect","bee","butterfly","dragonfly","wasp"]
    CLASSES_7A = ["bee","butterfly","dragonfly","wasp","other","insect","bumblebee"]
    CLASSES_7B = ["other","insect","bee","butterfly","dragonfly","wasp","bumblebee"]

    def __init__(self, root, class_names=CLASSES_6B, use_classes=None, use_samples=None):
        """
        Args:
            root (str): root path of dataset directory
            classes (_type_, list): Class list ordered by id, beginning at 0. Defaults to CLASSES_6B.
            use_classes (_type_, list): Load samples of only these classes. Defaults to None = all classes.
            use_samples (_type_, list): Only load these samples; Used to split in train and test with predefined lists.
        """
        self.root = root
        # <class_name>:<class_id>
        self.classes = dict(zip(class_names, range(len(class_names))))

        self.samples = []
        found_count = 0
        for f in Path(root).glob("*/*.csv"):
            found_count += 1
            if use_samples is not None:
                fn = f.name
                fid = frag_filename_to_id(fn)
                if fid not in use_samples:
                    continue
            clas = f.parent.name
            if use_classes is not None and clas not in use_classes:
                # skip this sample if class is not used
                continue
            point_set = np.loadtxt(f, delimiter=',', skiprows=1, usecols=(0,1,2)).astype(np.float32)
            # point_set = np.zeros((10,10))
            rel_path = str(Path(clas) / f.name)
            self.samples.append( (point_set, clas, rel_path) )
        print(f"Loaded dataset; {len(self.samples)} samples loaded; {found_count-len(self.samples)} samples skipped!")


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
    def get_class_list(args_classes):
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
        elif isinstance(args_classes, str) and "," in args_classes:
            classes = args_classes.lower().split(",")
        elif isinstance(args_classes, list):
            classes = args_classes
        else:
            raise RuntimeError("Unsupported classes: " + str(args_classes))
        return classes

    @staticmethod
    def load_dataset(dataset_dir, class_names, use_classes=None, batch_size=8, train_split=0.8, split_file=None):
        class_names = InsectDataLoader.get_class_list(class_names)
        if use_classes is None:
            use_classes = class_names
        use_classes = InsectDataLoader.get_class_list(use_classes)
        
        if split_file is not None:
            # Use predefined split for train and test samples; Read sample ids from files (one for train and test)

            # split_files is a string with a comma: "train_samples_file.txt,test_samples_file.txt"; in this order!
            # Must be in the dataset directory!
            train_test_fids = []
            split_file_path = Path(dataset_dir) / split_file
            print("Using train/test split file:", split_file_path)

            with open(split_file_path) as f:
                lines = f.read().splitlines()
                train_fids = [line.split(",")[-1] for line in lines if line.split(",")[0]=="train"]
                test_fids = [line.split(",")[-1] for line in lines if line.split(",")[0]=="test"]

            train_dataset = InsectDataLoader(root=dataset_dir, class_names=class_names, use_classes=use_classes, use_samples=train_fids)
            test_dataset = InsectDataLoader(root=dataset_dir, class_names=class_names, use_classes=use_classes, use_samples=test_fids)

            train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
            test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

            print("train, test size:", len(train_dataset), len(test_dataset))

        else:
            # Use percentual split; Or return the full dataset

            # dataset_dir = '../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-03_23-04-52'
            full_dataset = InsectDataLoader(root=dataset_dir, class_names=class_names, use_classes=use_classes)

            if train_split <= 0.0:
                # put all in test
                test_data_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
                print("train, test size:", 0, len(full_dataset))
                return class_names, use_classes, None, None, full_dataset, test_data_loader
            
            # else:
            # split in train and test
            train_size = int(train_split * len(full_dataset))
            test_size = len(full_dataset) - train_size
            # use fixed random generator
            g_cpu = torch.Generator()
            g_cpu.manual_seed(0)
            # split!
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=g_cpu)
            print("train, test size:", len(train_dataset), len(test_dataset))

            # data loaders
            train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
            test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
            
        return class_names, use_classes, train_dataset, test_dataset, train_data_loader, test_data_loader


if __name__ == '__main__':
    import torch
    full_dataset = InsectDataLoader('../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-23_12-17-56', 
                            class_names=InsectDataLoader.CLASSES_6B, use_classes=InsectDataLoader.CLASSES_6B)
    print("samples:", full_dataset.__len__())
    DataLoader = torch.utils.data.DataLoader(full_dataset, batch_size=4, shuffle=False, num_workers=1)
    
    # split in train and test
    train_split = 0.1
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    g_cpu = torch.Generator()
    g_cpu.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=g_cpu)
    print("train, test size:", len(train_dataset), len(test_dataset))

    # find number of samples per class
    print(len(train_dataset.indices))
    per_class = {}
    for ind in train_dataset.indices:
        pts,cid,pth = train_dataset.dataset._get_item(ind)
        # print(pth)
        clas = pth.split("\\")[0]
        l = per_class.setdefault(clas, [])
        l.append(pth)
    for clas in per_class:
        print(clas, "has", len(per_class[clas]), "samples")

    # get one batch (4 samples)
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
