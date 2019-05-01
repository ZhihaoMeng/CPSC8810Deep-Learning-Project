
from PIL import Image
import PIL
import os
import numpy as np

test_size = 0.2

for label in os.listdir('train_res'):
    for image in os.listdir('train_res'+'/'+label):
        im=Image.open('train_res'+'/'+label+'/'+image)
        im = im.convert('RGB')
        im = im.resize((224,224),PIL.Image.BILINEAR)
        if np.random.rand() < test_size:
            path = 'test_data'+'/'+label
            if not os.path.exists(path):
                if not os.path.exists('test_data'):
                    os.mkdir('test_data')
                os.mkdir(path)
            im.save(path+'/'+image)
        else:
            path = 'train_data'+'/'+label
            if not os.path.exists(path):
                if not os.path.exists('train_data'):
                    os.mkdir('train_data')
                os.mkdir(path)
            im.save(path+'/'+image)
            rm=im.rotate(5,PIL.Image.BILINEAR)
            rm.save(path+'/'+ 'r5'+image)
            rm=im.rotate(10,PIL.Image.BILINEAR)
            rm.save(path+'/'+ 'r10'+image)
            rm=im.rotate(-5,PIL.Image.BILINEAR)
            rm.save(path+'/'+ 'r-5'+image)
            rm=im.rotate(-10,PIL.Image.BILINEAR)
            rm.save(path+'/'+ 'r-10'+image)
            fm=im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            fm.save(path+'/'+ 'flip'+image)
