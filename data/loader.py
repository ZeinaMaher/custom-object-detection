import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    imgs, labels, filenames = zip(*batch)
    return torch.stack(imgs), labels, filenames

class CustomDataset(Dataset):
    def __init__(self, img_dir, labels_dir, shuffle, normalize, img_size=(480,480)):
        """
        img_dir : the folder that conatins the images 
        labels_dir : folder with annotation txt files (YOLO format)
        shuffle : True/False
        normalize: Whether to apply ImageNet normalization
        img_size: Target size for images(square)
        """
        self.img_dir= img_dir
        self.labels_dir= labels_dir
        self.img_size = img_size
        self.shuffle = shuffle
        self.normalize= normalize

        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg'))]

        #make sure each image has a txt file 
        self.img_files = [f for f in self.img_files 
                         if os.path.exists(os.path.join(self.labels_dir, f[:-4]+'.txt'))]
        if shuffle:
            random.shuffle(self.img_files)

        # ImageNet mean and std
        self.mean = torch.tensor([0.485, 0.456, 0.406]) if normalize else torch.tensor([0., 0., 0.])
        self.std = torch.tensor([0.229, 0.224, 0.225]) if normalize else torch.tensor([1., 1., 1.])

    def __len__(self):
        return len(self.img_files)

    def _get_label_path(self, file_name):
        label_path= os.path.join(self.labels_dir, file_name+ '.txt')
        return label_path

    def __getitem__(self, idx):
        file_name = self.img_files[idx][:-4] # handle only jpg
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path= self._get_label_path(file_name)

        # load image and annotation
        img = Image.open(img_path).convert('RGB')
        with open(label_path , 'r')as f:
            lines= f.read().strip().split('\n')
        labels = [list(map(float, line.split(' '))) for line in lines]

        # preprocessing
        img, labels= self.preprocess(img, labels)
        return img, labels ,file_name  

    def _resize_image(self, img):
        return img
    
    def _to_tensor(self, img, labels):
        """convert numpy arrays to tensors"""
        img = img.transpose((2, 0, 1)) #swap color axis (HWC to CHW)
        img= torch.from_numpy(img)

        labels= torch.from_numpy(labels)
        return img, labels

    def preprocess(self, img, labels):
        # Convert to numpy arrays
        img = np.array(img)
        labels = np.array(labels) if labels else np.zeros((0, 5))  # Empty array if no labels
        
        img= self._resize_image(img)
        
        img, labels= self._to_tensor(img, labels)

        # Normalize
        img= img/255.0
        if self.normalize:
            img = (img - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        return img, labels.float()

if __name__ == "__main__":
    dataset= CustomDataset(r'C:\Users\Zeina Abu Ruqaia\Desktop\projects\Custom_Object_Detector\dataset\filtered_data\test\images',
                        r'C:\Users\Zeina Abu Ruqaia\Desktop\projects\Custom_Object_Detector\dataset\filtered_data\test\labels',
                        True, True)

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )