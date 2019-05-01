
from PIL import Image
import PIL
import os
for label in os.listdir('train'):
    for image in os.listdir('train'+'/'+label):
        im=Image.open('train'+'/'+label+'/'+image)
        im = im.convert('RGB')
        im = im.resize((224,244), PIL.Image.BILINEAR)
        im.save('train'+'/'+label+'/'+image)
       # rm=im.rotate(5)
       # rm.save('train'+'/'+label+'/'+ 'r5'+image)
       # rm=im.rotate(10)
       # rm.save('train'+'/'+label+'/'+ 'r10'+image)
       # rm=im.rotate(-5)
       # rm.save('train'+'/'+label+'/'+ 'r-5'+image)
       # rm=im.rotate(-10)
       # rm.save('train'+'/'+label+'/'+ 'r-10'+image)
       # fm=im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
       # fm.save('train'+'/'+label+'/'+ 'flip'+image)"""
