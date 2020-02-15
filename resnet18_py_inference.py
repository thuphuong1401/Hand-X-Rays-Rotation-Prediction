import torch
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms, utils, models

# define dataset class for data loader
class HandXRayDataset(Dataset):
    """HandXRayDataset dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        #get rid of this
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = np.asarray(image) / 255.0
        image = np.transpose(image)
        image = image.astype(np.float32)
        #print(type(image))
        landmarks = self.landmarks_frame.iloc[idx,1]
        # landmarks = landmarks.astype('float').reshape(-1, 1)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# identify where the weights you want to load are
weight_fil = "Model 1/best_weights.pth"

# identify where the data you want to test on is using a command line argument
#data_fil = "/"

# set necessary hyperparameters
batch_size = 10
loss_func = nn.MSELoss()

# initialize model
model = models.resnet18(pretrained = False)
num_final_in = model.fc.in_features
NUM_CLASSES = 1
model.fc = nn.Linear(num_final_in, NUM_CLASSES)

# # load weights
model = torch.load(weight_fil)

# put model in evaluation mode (sets dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results)
model.eval()

# create loaders to feed in data to the network in batches
eval_set = HandXRayDataset(csv_file = "test_labels.csv", root_dir = 'test')
eval_loader = torch.utils.data.DataLoader( dataset = eval_set , batch_size= batch_size , shuffle = True)

# track metrics over dataset
eval_loss = 0.0

# run testing using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loop through eval data
for i, sample in enumerate(eval_loader):

    images = sample['image']
    labels = sample['landmarks']

    images = images.to(device)
    labels = labels.to(device)

	# run the model on the eval batch
    outputs = model(images)

	# compute eval loss
    loss = loss_func(outputs, labels)
    eval_loss += loss.item()

print("Loss = " + str(eval_loss/(len(eval_set)/batch_size)))
