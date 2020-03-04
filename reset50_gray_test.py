import os
import torch
import torchvision
import torch.nn as nn 
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T 
import numpy as np
import time
import cv2

class MyResNet(nn.Module):

    def __init__(self, in_channels=1):
        super(MyResNet, self).__init__()

        # bring resnet
        self.model = torchvision.models.resnext50_32x4d(pretrained=True)
        #self.model = list(self.model.children())
        w = self.model.conv1.weight
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        #self.model.fc = torch.nn.Linear(512 * self.model.layer1[0].expansion, 2)
        #self.model = nn.Sequential(*self.model)

        # original definition of the first layer on the renset class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # your case
        #self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.model.fc = torch.nn.Linear(512 * self.model.layer1[0].expansion, 1)

        '''
        arch = models.resnet50(num_classes=1000, pretrained=True)
        arch = list(arch.children())
        w = arch[0].weight
        arch[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
        arch[0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        arch = nn.Sequential(*arch)
        '''

    def forward(self, x):
        return self.model(x)

def Preprocess():
    '''We can then read the input test image, resize it such that the smaller size is 224, 
    preserving the aspect ratio while resizing. The center 224×224 image is cropped out 
    and converted into a tensor. This step converts the values into the range of 0-1. 
    It is then normalized relative to the ImageNet color cluster using the ImageNet mean 
    and standard deviation. It is done as input[channel] =(input[channel] – mean[channel]) / std[channel].
    '''
    transform = T.Compose([            #[1]
    T.Resize(256),                    #[2]
    T.CenterCrop(224),                #[3]
    T.ToTensor(),                     #[4]
    T.Normalize(                      #[5]
    #mean=[0.485, 0.456, 0.406],                #[6]
    #std=[0.229, 0.224, 0.225]                  #[7]
    mean=[0.5],
    std=[0.5]
    )])

    return transform

def Inference(net, image_path, labels):

    img = Image.open(image_path)
    #b, g, r = img.split()
    #rgb_img = Image.merge("RGB", (r, g, b))
    if img.mode == 'RGB':
        gray_img = img.convert("L")

    #plt.imshow(img); 
    plt.imshow(gray_img)
    plt.axis('off'); 
    plt.show()
    
    '''
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray image", gray_img)
    cv2.waitKey(0)

    pil_gray_img = Image.fromarray(gray_img)
    plt.imshow(pil_gray_img)
    plt.show()
    '''

    start_time = time.time()
    img_t = Preprocess()(gray_img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = net(batch_t)
    end_time = time.time()
    print("take %f seconds" % (end_time - start_time))
    #print(out.sjhape)

    #with open(classes_txt) as f:
    #    labels = [line.strip() for line in f.readlines()

    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    for idx in indices[0][:5]:
        print(labels[idx], percentage[idx].item())
    #[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
    #print(labels[indices[0]], percentage[indices[0]].item())

    
    


if __name__ == '__main__':
    resnet50 = MyResNet().eval()
    #resnet50 = models.resnext50_32x4d(pretrained=True).eval()
    #fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
    abs_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(abs_path,'dog.jpg')

    classes_txt = os.path.join(abs_path, 'imagenet_classes.txt')
    labels=[]
    with open(classes_txt) as f:
        labels = [line.strip() for line in f.readlines()]

    Inference(resnet50, image_path, labels)

    