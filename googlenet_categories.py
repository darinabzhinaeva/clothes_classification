# import libs
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet34
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as opt
import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
train_transforms = T.Compose(
    [
     T.Resize((224, 224), interpolation=3),
     T.ColorJitter(), # изменение цвета
     T.RandomHorizontalFlip(p=0.2), # случайное горизонтальное переворачивание
     T.RandomRotation(20), # вращение
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

#Test Time Augmentation
test_transforms = T.Compose(
    [
     T.Resize((224, 224), interpolation=3),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
train_set = datasets.ImageFolder('gender_train', train_transforms)
# train_set2 = datasets.ImageFolder('/content/gdrive/My Drive/dataset/car_human_noise/train', train_transforms2)
# train_set = train_set1 + train_set2
test_dataset = datasets.ImageFolder('gender_test', test_transforms)
#separate train data to train and valid datasets
train_size = int(1 * len(train_set))
valid_size = len(train_set) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_set, [train_size, valid_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                           pin_memory=True, shuffle = True,
                                           num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20,
                                          num_workers=0)

print("train_sampler: ", len(train_dataset))
print("valid_sampler: ", len(valid_dataset))
print("test_sampler: ", len(test_dataset))

class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch4]#branch2 branch3
        return torch.cat(outputs, 1)

class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        print("\n !!!!!!!!!inception \n")

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)

        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=False, transform_input=False, init_weights=True):#init_weights=True
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3) #224->112
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True) #112->55
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True) #55->27

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        #                    in_channels, 1out,  2out, 2out, 3out, 3out,  4out
        # self.inception3b = Inception(128, 128,   128,  192,  32,   96,    64)#256
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True) #27->13

        # self.inception4a = Inception(288, 192, 96, 208, 16, 48, 64)#480
        # for delete inception 3b change inputs inception 4a
        self.inception4a = Inception(96, 192, 96, 208, 16, 48, 64)#480 128
        # self.inception4b = Inception(304, 160, 112, 224, 24, 64, 64)#512
        # self.inception4c = Inception(288, 128, 128, 256, 24, 64, 64)#512

        # for delete inception 4b, change input inception 4c
        self.inception4c = Inception(256, 128, 128, 256, 24, 64, 64)#512 304
        self.inception4d = Inception(192, 112, 144, 288, 32, 64, 64)#512 256
        # self.inception4e = Inception(240, 256, 160, 320, 32, 128, 128)#528
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) #13->6

        # self.inception5a = Inception(512, 256, 160, 320, 32, 128, 128)#832

        # for delete inception 4e, change input inception 5a
        # self.inception5a = Inception(240, 256, 160, 320, 32, 128, 128)#832


        # self.inception5b = Inception(512, 384, 192, 384, 48, 128, 128)#832
        self.inception5b = Inception(176, 384, 192, 384, 48, 128, 128)#832 240
        # if aux_logits:
        #     self.aux1 = InceptionAux(512, num_classes)
        #     self.aux2 = InceptionAux(528, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #6->1
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)#1024 640

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        # x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        # if self.training and self.aux_logits:
        #     aux1 = self.aux1(x)

        # x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        # if self.training and self.aux_logits:
        #     aux2 = self.aux2(x)

        # x = self.inception4e(x)
        x = self.maxpool4(x)
        # x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        # if self.training and self.aux_logits:
        #     return aux1, aux2, x
        return x


PATH = "mod.pth"
model = resnet34(3)
# model = GoogLeNet(3)
# model = GoogLeNet(3)
# model = model.cuda()

optimizer = opt.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss().cuda()

max_epoch = 7
metric_valid = []
best_result = 0.0

for epoch in range(max_epoch):
    model.train()
    print("epoch : ", epoch)
    for iteration, (train_dataset, labels) in enumerate(train_loader):
        # train_dataset, labels = train_dataset.cuda(), labels.cuda()
        # images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(train_dataset)  # forward
        loss = criterion(outputs, labels)  # without Aux only 1 output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trueth = 0
    total = 0
    for it, (test_dataset, labels) in enumerate(test_loader):
        # test_dataset, labels = test_dataset.cuda(), labels.cuda()
        outputs = model(test_dataset)
        _, preds = torch.max(outputs, 1)
        trueth += (
                    preds == labels).sum().item()  # accuracy_score(np.array(labels.cpu()), np.array(preds.cpu()), normalize=False)
        total += len(labels)
    print("Accuracy test = ", str(trueth / total))
    # print(loss.item())
    # //torch.save(model.state_dict(), f'{ep}.pth')
    # //model.load_state_dict(torch.load('model.pth'))
    # print("Validation...")
    # model.eval()
    # Trueth = 0
    # total = 0
    # for iteration, (valid_dataset, labels) in enumerate(valid_loader):
    #   valid_dataset, labels = valid_dataset.cuda(), labels.cuda()
    #   outputs = model(valid_dataset) # forward
    #   #outputs = (outputs2[0] + outputs2[1] + outputs2[2])/3
    #   _, preds = torch.max(outputs, 1)
    #   #loss = criterion(outputsV, labels)
    #   #Trueth += accuracy_score(np.array(labels.cpu()), np.array(preds.cpu()), normalize=False)
    #   Trueth += (preds == labels).sum().item()
    #   total += len(labels)
    # print(loss.item())
    # result = (Trueth/total)
    # print("Accuracy valid = ", str(result))
    # metric_valid.append(result)
    # if result > best_result:
    #   print("find best model in ", epoch, " epoch")
    #   torch.save({
    #           'epoch': epoch,
    #           'model_state_dict': model.state_dict(),
    #           'optimizer_state_dict': optimizer.state_dict(),
    #           'loss': loss
    #           }, PATH)
    #   #best_model = model #https://pytorch.org/tutorials/beginner/saving_loading_models.html
    #   best_result = result

    # //print(f'Accuracy on validation = {truth / len(valid_dataset)}')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, PATH)
