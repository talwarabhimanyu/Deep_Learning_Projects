from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv
import re
import shutil
import os

# This module contains utilities for loading and processing data.
class SingleLabelImages(Dataset):
    @classmethod
    def _parseCSV(cls, mode, data_path, csvfile, has_header, classes, max_read):
        idx_to_img = {}
        idx_to_label = {}
        if 'test' in mode:
            for i, img_name in enumerate(os.listdir(data_path)):
                idx_to_img.update({i : re.sub('.[a-zA-Z]', '', img_name)})
        else:
            with open(csvfile, 'r') as f:
                reader = iter(csv.reader(f))
                if has_header:
                    row = next(reader)
                for i, row in enumerate(reader):
                    if (i == max_read):
                        break
                    idx_to_img.update({i : row[0]})
                    try:
                        idx_to_label.update({i : classes.index(row[1])})
                    except:
                        print(row[1])
        return idx_to_label, idx_to_img
    def __init__(self, mode, data_path, csvfile, classes, file_ext, transform=None, max_read=0, has_header=True):
        self.mode = mode
        self.data_path = data_path
        self.classes = classes
        self.num_classes = len(classes)
        self.file_ext = file_ext
        self.transform = transform
        self.csvfile = csvfile
        self.max_read = max_read
        self.has_header = has_header
        self.idx_to_label, self.idx_to_img = self._parseCSV(self.mode, self.data_path, 
                                                      self.csvfile, self.has_header, self.classes, self.max_read)
        print('Read {} {} images.'.format(len(self.idx_to_img), self.mode))
    def __len__(self):
        return len(self.idx_to_img)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.idx_to_img[idx] + self.file_ext)
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            if self.transform is not None:
                image = self.transform(image)
            label = ''
            if not('test' in self.mode):
                label = self.idx_to_label[idx]
            return {'image' : image, 'label' : label, 
                    'id' : self.idx_to_img[idx]}

# Split data into train and validation set
def SplitTrainValid(data_dir, train_dir, valid_dir, data_file, train_file, valid_file, valid_percent, has_header):
    progress_bar = tqdm(total=50000)
    with open(train_file, 'w') as train_f:
        train_writer = csv.writer(train_f, lineterminator='\n')
        with open(valid_file, 'w') as valid_f:
            valid_writer = csv.writer(valid_f, lineterminator='\n')
            with open(data_file, 'r') as f:
                reader = iter(csv.reader(f))
                if has_header:
                    row = next(reader)
                    train_writer.writerow(row)
                    valid_writer.writerow(row)
                for row in reader:
                    rand_num = np.random.uniform(0, 1)
                    if (rand_num > valid_percent/100.0):
                        train_writer.writerow(row)
                        shutil.copyfile(os.path.join(data_dir, row[0]+'.png'),
                                       os.path.join(train_dir, row[0]+'.png'))
                    else:
                        valid_writer.writerow(row)
                        shutil.copyfile(os.path.join(data_dir, row[0]+'.png'),
                                       os.path.join(valid_dir, row[0]+'.png'))
                    progress_bar.update(1)
