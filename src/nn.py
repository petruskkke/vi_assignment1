from __future__ import division

import os
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import randint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models


logger = logging.getLogger('main')


args = {
    'trainset': 'canvas/Assignment1_data/data',
    'valset': 'canvas/Assignment1_data/query',
    'resume': None,

    'imsize': 224,
    'batch_size': 16,
    'epochs': 20, 
    
    'topn': 5,
}


def nnruner(train=False, pretrained=True):
    model, train_loader, val_loader, optimizer, lr_scheduler, start_epoch = preppare(pretrained)
    mse_criterion = MeanSquareLoss(1e-8)
    
    if train and pretrained:
        return

    if not train:
        val(model, train_loader, val_loader, pretrained)
        return 

    for epoch in range(start_epoch, args['epochs']):
        model.train()
        for idx, (images, names) in enumerate(train_loader):
            optimizer.zero_grad()

            images = Variable(images).cuda()

            predicts, _ = model.forward(images)        

            loss = mse_criterion(predicts, images)
            loss.mean().backward()        
            optimizer.step()

            logger.info('step: [{0}][{1}/{2}]\t loss: {3}'.format(
                epoch, idx, len(train_loader), torch.mean(loss.data)))
            
        lr_scheduler.step()
        val(model, train_loader, val_loader, pretrained, epoch)


def val(model, train_loader, val_loader, pretrained, epoch=0):
    name_list = []
    dist_dict = {}
    model.eval()
    for idx, (valims, val_names) in enumerate(val_loader):
        valims = Variable(valims).cuda()
        if pretrained:
            val_features = model.forward(valims)
        else:
            _, val_features = model.forward(valims)

        val_features = val_features.data.cpu().numpy()
        for name in val_names:
            dist_dict[name] = []

        for jdx, (ims, train_names) in enumerate(train_loader):
            ims = Variable(ims).cuda()
            
            if pretrained:
                train_features = model.forward(ims)
            else:
                _, train_features = model.forward(ims)

            train_features = train_features.data.cpu().numpy()

            for tdx, tfeature in enumerate(train_features):
                for vdx, vfeature in enumerate(val_features):
                    dist = np.sqrt(np.sum((tfeature - vfeature) ** 2))

                    dist_dict[val_names[vdx]].append(dist)
                name_list.append(train_names[tdx])

        rows, cols = 5, args['topn'] + 1
        fig, ax = plt.subplots(rows, cols)

        idx = 0
        for key, item in dist_dict.items():
            query = key
            idx = int(query.split('/')[-1].split('.')[0])
            dists = np.asarray(item)

            ax[idx, 0].axis('off')
            ax[idx, 0].imshow(np.asarray(Image.open(query)))

            jdx = 1
            dmax = np.max(dists)
            for tdx in range(0, args['topn']):
                imin = np.argmin(dists)
                ax[idx, jdx].axis('off')
                ax[idx, jdx].imshow(np.asarray(Image.open(name_list[imin])))
                dists[imin] = dmax
                jdx += 1
            # idx += 1
        plt.savefig('results/p4a2_{0}.jpg'.format(epoch), bbox_inches='tight')
            

def preppare(pretrained):
    if pretrained:
        model = models.resnet50(pretrained=True)
    else:
        model = Model()
    model = nn.DataParallel(model).cuda()
    logger.info('model: \n')
    logger.info(model_str(model))

    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    trainset = ImDataset(args['trainset'], args['imsize'], 
        transform=transform, fliplr=True)
    valset = ImDataset(args['valset'], args['imsize'], 
        transform=transform, fliplr=False)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], 
            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=5, shuffle=False,
        num_workers=2, pin_memory=True)

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                            # lr=1e-2, momentum=0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))  
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=int(args['epochs'] / 4), gamma=0.1)

    start_epoch = 0
    if args['resume'] is not None:
        logger.info('load checkpoint: ' + args['resume'])
        checkpoint = torch.load(args['resume'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, train_loader, val_loader, optimizer, lr_scheduler, start_epoch


def model_str(module):
    lines = [
        "",
        "model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} = {total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(
            row_format.format(
                name=name,
                shape=" * ".join(str(p) for p in param.size()),
                total_size=param.numel()))
    lines.append("=" * 75)
    lines.append(
        row_format.format(
            name="all parameters",
            shape="sum of above",
            total_size=sum(int(param.numel()) for name, param in params)))
    lines.append("")
    return "\n".join(lines)


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(EncoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pooling(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.out_channel = out_channel

    def forward(self, x, y=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if y is not None:
            x = torch.cat((x, y), 1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.enc2 = EncoderBlock(8, 32, stride=2)      
        self.enc3 = EncoderBlock(32, 64, stride=2)  
        self.enc4 = EncoderBlock(64, 128, stride=2)  

        self.dec1 = DecoderBlock(128, 64)          
        self.dec2 = DecoderBlock(64, 32)               
        self.dec3 = DecoderBlock(32, 8)               
        self.final= nn.Conv2d(8, 3, 1)                          

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec1 = self.dec1(enc4)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        final = self.final(dec3)

        return final, enc4


class ImDataset(Dataset):
    def __init__(self, path, im_size, transform=None, fliplr=True):
        self.path = path
        self.transform = transform
        self.fliplr = fliplr
        self.im_size = im_size
        self.im_list = [name for name in os.listdir(self.path)]

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        name = os.path.join(self.path, self.im_list[idx])

        image = Image.open(name)

        if image.width != self.im_size or image.height != self.im_size:
            # resize
            if image.width < image.height:
                rewidth, reheight = self.im_size, image.height * self.im_size // image.width
            else:
                rewidth, reheight = image.width * self.im_size // image.height, self.im_size
            image = self._resize(image, (rewidth, reheight))
            # crop
            sx, sy = (rewidth - self.im_size) // 2, (reheight - self.im_size) // 2
            ex, ey = sx + self.im_size, sy + self.im_size
            image = image.crop((sx, sy, ex, ey))

        image = np.array(image).astype('float32') / 255.0

        if self.fliplr and (randint(0, 1) == 0):
            image = np.fliplr(image).copy()

        if self.transform:
            image = Image.fromarray((image * 255.0).astype('uint8'))
            image = self.transform(image)

        return image, name

    def _resize(self, img, size):
        if size[0] > img.width:
            interp = cv2.INTER_LINEAR
        else:
            interp = cv2.INTER_AREA

        return Image.fromarray(
            cv2.resize(np.array(img).astype('uint8'), size, interpolation=interp))


class MeanSquareLoss(nn.Module):
    def __init__(self, epsilon):
        super(MeanSquareLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        loss = torch.sqrt((pred - gt) ** 2 + self.epsilon)
        return loss.mean(1).mean(1)
