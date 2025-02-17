import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch

class VideoRecord(object):
    def __init__(self, root_path, vid_len, vid_id):
        self._data = [os.path.join(root_path, vid_id + '_' + str(frame_id) + '.png') for frame_id in range(vid_len)]
        # example: 6-th video contains list self._data = root_path + [5_0, 5_1, ..., 5_{vid_len-1}]

    @property
    def paths(self):
        return self._data


class DataSetSegVid(data.Dataset):
    def __init__(self, root_path, list_file, video_len, modality='RGB', transform=None, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.modality = modality

        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.vid_len = video_len

        self._parse_list()

    def _load_image(self, path):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return Image.open(path).convert('L')
        elif self.modality == "Flow":
            return Image.open(path).convert('L')

    def _parse_list(self):
        list_files = open(self.list_file,'r').read().splitlines()
        self.video_list = [VideoRecord(self.root_path, self.vid_len, vid_id) for vid_id in list_files]

    def __getitem__(self, index):
        record = self.video_list[index]
        one_video = self.get(record)
        return one_video  

    def get(self, record):
        a = np.random.randint(0, 4)
        inp_imgs = [self._load_image(path) for path in record.paths]

        process_data = [self.transform([inp_img, a]) for inp_img in inp_imgs]
        seg_imgs = [self._load_image(path.replace('imgs', 'gt')) for path in record.paths]
        process_label = [self.transform([seg_img, a]) for seg_img in seg_imgs]

        return process_data, process_label

    def __len__(self):
        return len(self.video_list)