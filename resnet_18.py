from __future__ import print_function, division
import os
import torch
import pandas as pd
#from scikit-image import io, transform
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch import nn, optim
from datetime import datetime


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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

face_dataset = HandXRayDataset(csv_file='train_labels.csv', root_dir='train')

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    #print(sample)

    #print(i, sample['image'], sample['landmarks'].shape, sample['landmarks'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_func = nn.MSELoss()
loss_func_name = 'mean squared error loss'
learn_rate = .001
num_epochs = 20
batch_size = 10
#confidence_threshold = 0.5
#loss_adj_conf_thresh = np.log(confidence_threshold)
optimizer_name = 'Adam'
start_time = datetime.now()

# This folder is where I save all relevent files to (hyperparams file, training logs, model weights, etc.)
run_id = "Model 1"
os.mkdir(run_id)

# record all hyperparameters that might be useful to reference later
with open(run_id + '/hyperparams.csv', 'w') as wfil:
	#wfil.write("activation function," + "ReLU" + '\n')
	wfil.write("loss function," + loss_func_name + '\n')
	wfil.write("learning rate," + str(learn_rate) + '\n')
	wfil.write("number epochs," + str(num_epochs) + '\n')
	wfil.write("batch size," + str(batch_size) + '\n')
	wfil.write("optimizer," + str(learn_rate) + '\n')
	wfil.write("start time," + str(start_time) + '\n')


# create loaders to feed in data to the network in batches
train_set = face_dataset
trainloader = torch.utils.data.DataLoader(dataset = train_set , batch_size= batch_size , shuffle = True)
valid_set = HandXRayDataset(csv_file = "valid_labels.csv", root_dir = 'valid')
validloader = torch.utils.data.DataLoader(dataset = valid_set , batch_size= batch_size , shuffle = True)


# initialize network
model = models.resnet18(pretrained = True)
num_final_in = model.fc.in_features
NUM_CLASSES = 1
model.fc = nn.Linear(num_final_in, NUM_CLASSES)

# Code for freezing different layers
ct = 0
for child in model.children():
    ct += 1
    if ct < 4:
        for param in child.parameters():
            param.require_grad = False


# send the model to GPU
model.to(device)

# initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

# track best val loss to know when to save best weights
best_valid_loss = "unset"

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

with open(run_id + '/log_file.csv', 'w') as log_fil:
	# write headers for log file
    log_fil.write("epoch,epoch duration,train loss,valid loss\n")
    for epoch in range(0, num_epochs):
        epoch_start = datetime.now()
        # track train and validation loss
        epoch_train_loss = 0.0
        epoch_valid_loss = 0.0
        for i, sample in enumerate(trainloader):
            images = sample['image']
            labels = sample['landmarks']

            images = images.to(device)
            labels = labels.to(device)

			# zero out gradients for every batch or they will accumulate
            optimizer.zero_grad()

            #print(images)

			# forward step
            outputs = model(images)

			# compute loss
            loss = loss_func(outputs, labels)

			# backwards step
            loss.backward()

			# update weights and biases
            optimizer.step()

			# track training loss
            epoch_train_loss += loss.item()

		# track valid loss - the torch.no_grad() unsures gradients will not be updated based on validation set
        with torch.no_grad():
            for i, sample in enumerate(validloader):
                images = sample['image']
                labels = sample['landmarks']
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_func(outputs, labels)
                epoch_valid_loss += loss.item()


		# track total epoch time
        epoch_end = datetime.now()
        epoch_time = (epoch_end - epoch_start).total_seconds()

		# save best weights
        if best_valid_loss=="unset" or epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model, run_id + "/best_weights.pth")

		# save most recent weights
        torch.save(model, run_id + "/last_weights.pth")

		# save epoch results in log file
        log_fil.write( str(epoch) + ',' + str(epoch_time) + ',' + str(epoch_train_loss/(len(train_set)/batch_size)) + ',' + str(epoch_valid_loss/(len(valid_set)/batch_size)) + '\n' )

		# print out epoch level training details
        print("epoch: " + str(epoch) + " - ("+ str(epoch_time) + " seconds)" + "\n\ttrain loss: " + str(epoch_train_loss/(len(train_set)/batch_size)) + "\n\tvalid loss: " + str(epoch_valid_loss/(len(valid_set)/batch_size)))

end_time = datetime.now()
with open(run_id + '/hyperparams.csv', 'a') as wfil:
	wfil.write("end time," + str(end_time) + '\n')
