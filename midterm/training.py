import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from VGG import VGG
from Inception import inception_v3
import torch.nn as nn
import numpy as np
import warnings
import PIL
warnings.filterwarnings("ignore")


img_size = (224,224)
#img_size = (256, 256)
crop_size = 224
train_transforms = transforms.Compose([
   # transforms.RandomAffine(20, resample=PIL.Image.BILINEAR),
   # transforms.Resize(img_size),
    #transforms.RandomSizedCrop(crop_size),
    #transforms.RandomHorizontalFlip(0.5),
    #transforms.RandomCrop(crop_size),
   # transforms.RandomVerticalFlip(0.5),
   # transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    #transform.Pad(padding=2,fill=(r,g,b)),
    #transforms.Grayscale(num_output_channels=1),
    #transforms.ColorJitter(saturation=0.05,hue=0.05),
    #transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5761, 0.5314, 0.5042), (0.2501, 0.2501, 0.2506))

])

test_transforms = transforms.Compose([
   # transforms.Resize(img_size),
    #transforms.RandomHorizontalFlip(0.5),
    #transforms.RandomCrop(crop_size),
    #transform.Pad(padding=2,fill=(r,g,b)),
    #transforms.Grayscale(num_output_channels=1),
    #transforms.ColorJitter(saturation=0.05,hue=0.05),
    #transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5761,0.5314,0.5042), (0.2501, 0.2501, 0.2506))
])

path ='train'
valid_size = 0.3
#def load_split_train(path, valid_size = .2,transforms):
train_set = ImageFolder(root = 'train_data',transform=train_transforms)
test_set = ImageFolder(root = 'test_data',transform=test_transforms)
#train_set = ImageFolder(root = 'data/train', transform = train_transforms)
#test_set = ImageFolder(root = 'data/test', transform = test_transforms)
num_train = len(train_set)
#indices = list(range(num_train))
#split = int(np.floor(valid_size * num_train))
#np.random.seed(47)
#np.random.shuffle(indices)
#train_idx, test_idx = indices[split:],indices[:split]
#train_sampler = SubsetRandomSampler(train_idx)
#test_sampler = SubsetRandomSampler(test_idx)
#total_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,num_workers=4)
#train_loader = DataLoader(train_set, batch_size=32, shuffle = True,num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False,num_workers=4)
#test_loader = DataLoader(test_set, batch_size=32,shuffle = False, num_workers=4)
"""mean = 0
std = 0
for images, _ in total_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean +=images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /=len(total_loader.dataset)
std /=len(total_loader.dataset)
print(mean)
print(std)"""   	
num_epochs = 300
learning_rate = 0.00005

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

model = nn.DataParallel(VGG(in_channels=3),[0,1]).cuda()
#model = nn.DataParallel(inception_v3(pretrained=False,aux_logits=True),[0,1]).cuda()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.01)
#optimizer = torch.optim.SGD(model.parameters(),learning_rate,momentum=0.9,weight_decay=1e-4)
exp_lr_scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,\
gamma=0.9)
scheduler = exp_lr_scheduler

# Train the model
total_step = len(train_set)
#total_step = len(train_set)
for epoch in range(num_epochs):
   # np.random.seed()
    model.train(True)
    train_loss = 0
    correct = 0
    total = 0
    scheduler.step()
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Run the forward pass
       # outputs, aux = model(images)
       # loss = 1.0*criterion(outputs, labels)+0.4*criterion(aux,labels)
		
        outputs = model(images)
        loss = criterion(outputs,labels)
        # loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
	#train_loss += loss.item()

        # model.eval()
   # Track the accuracy
        total += labels.size(0)
   # total =  i + 1
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    #print(predicted.item(),"   ",labels.item(),'\n')
    # acc_list.append(correct / total)

    if total == total_step:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
   	 .format(epoch + 1, num_epochs, total, total_step, loss.item(), (correct / total) * 100))
   # train_loss=0
    #intermediate test
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of Epoch {} on the  test images: {} %'.format(epoch + 1, (correct / total) * 100))
    acc = (correct/total)*100
    if acc>65:
        torch.save(model.state_dict(), 'VGG19_split_02'+str(acc)+'%.ckpt')
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #test[labels.item()] += labels.size(0)
        #res_correct[labels.item()] += (predicted==labels).sum().item()
    print('Test Accuracy of the model on the  test images: {} %'.format((correct / total) * 100))
    acc = (correct/total)*100
    if acc>85:
        torch.save(model.state_dict(), 'VGG19__mode_'+str(acc)+'%.ckpt')
print("test_set_labels:", test_set.class_to_idx, "\n")
#print("train_set_labels:",train_set.class_to_idx,"\n")
#print("total test images:", test,"\n")
#print("total correct image:", res_correct, "\n")

# save the labels for later use
import json
with open('label.json','w') as outfile:
    json.dump(train_set.classes,outfile)

# save the model
torch.save(model.state_dict(),'VGG19__mode.ckpt')
