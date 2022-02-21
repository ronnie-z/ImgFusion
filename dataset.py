import logging
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, vis_root_path, transform=None):
        super(DataSet, self).__init__()
        self.vis_root_path = vis_root_path
        self.ir_root_path = vis_root_path.replace('vis', 'ir')
        self.transform = transform
        self.images = list(os.listdir(vis_root_path))

    def __getitem__(self, index):
        vis_path = os.path.join(self.vis_root_path, self.images[index])
        ir_path = os.path.join(self.ir_root_path, self.images[index].replace('VIS', 'IR'))
        vis_img = cv2.imread(vis_path, flags = 0)[:, :, np.newaxis]  # vis图像  1 H W
        ir_img = cv2.imread(ir_path, flags = 0)[:, :, np.newaxis]    # ir图像
        if self.transform is not None:
            vis_img = self.transform(vis_img)
            ir_img = self.transform(ir_img)
        return vis_img, ir_img    
    
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    pass