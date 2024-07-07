'''
@author: Xu Yan, Marc Böge
'''
import os
import numpy as np
import warnings
import pickle
from pathlib import Path

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

class InsectDataLoader(Dataset):
    CLASSES_4 = ["bee","butterfly","dragonfly","wasp"]
    CLASSES_5 = ["bee","butterfly","dragonfly","wasp","insect"]
    CLASSES_6 = ["bee","butterfly","dragonfly","wasp","insect","other"]

    def __init__(self, root, classes=CLASSES_5):
        self.root = root
        # <class_name>:<class_id>
        self.classes = dict(zip(classes, range(len(classes))))

        self.samples = []
        for f in Path(root).glob("*/*.csv"):
            clas = f.parent.name
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


if __name__ == '__main__':
    import torch
    data = InsectDataLoader('../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_2024-07-03_23-04-52')
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
