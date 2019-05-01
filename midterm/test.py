from PIL import Image
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision import transforms
import sys
from vgg import vgg19_bn
from VGG import VGG
#from Inception import Inception3
#image = Image.open(sys.argv[1])
#x = TF.resize(image,(224,224)
#x = TF.to_tensor(x)
classes = ["gossiping", "isolation", "laughing", "nonbullying", "pullinghair", "punching", "quarrel", "slapping", "stabbing", "strangle"]

img_size = (224,224)
loader = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))] )

def image_loader(loader,image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image.cuda()  #assumes that you're using GPU

# load model and classify the image
#model = nn.DataParallel(vgg19_bn(num_classes=10),[0,1]).cuda()
model = nn.DataParallel(VGG(),[0,1]).cuda()
#model = VGG()
model.load_state_dict(torch.load('VGG19_split_0271.54471544715447%.ckpt'))
#model.eval()
#if torch.cuda.device_count()>1:
#    model = nn.DataParallel(model,[0,1]).cuda()
#else:
#model = model.cuda()
# return the result
#image = Variable(image).cuda()
model.eval()
image = image_loader(loader,sys.argv[1])
output = model(image)
_, predicted = torch.max(output.data,1)
#print(predicted.item())
res = predicted.item()

print(classes[res])
