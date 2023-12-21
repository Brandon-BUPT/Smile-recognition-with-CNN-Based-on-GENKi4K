import glob
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms


def read_column(file_path, i):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        column = [float(line.strip().split(' ')[i]) for line in lines]
        if i == 3:
            column = [float(line.strip().split(' ')[i].replace("\n", "")) for line in lines]

    return column


class Mydataset(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, item):
        img = self.imgs[item]
        label = self.labels[item]
        pil_img = Image.open(img)
        pil_img = pil_img.convert("RGB")
        data = self.transforms(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)


class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def fit(epoch, model,loss_fn, optim, train_dl, test_dl):
    train_loss = 0
    model.train()
    for x, y in train_dl:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y.float())
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            train_loss += loss.item()
    train_loss = train_loss / len(train_dl.dataset)

    test_running_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_running_loss += loss.item()
    test_loss = test_running_loss / len(test_dl.dataset)
    print('epoch: ', epoch, 'train_loss： ', round(train_loss, 3), 'test_loss： ', round(test_loss, 3),)


if __name__ == "__main__" :
    imagePaths = glob.glob(".\\Data\\genki4k\\files\\*.jpg")
    # Which parameter to predict   i =1~3
    i = 1
    labelPaths = read_column(".\\Data\\genki4k\\labels.txt", i)
    print(labelPaths)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    index = np.random.permutation(len(imagePaths))
    all_imgs_path = np.array(imagePaths)[index]
    all_labels = np.array(labelPaths)[index]
    print(all_labels)
    print(all_imgs_path)

    s = int(len(all_imgs_path) * 0.8)
    train_imgs = all_imgs_path[:s]
    train_labels = all_labels[:s]
    val_imgs = all_imgs_path[s:]
    val_labels = all_labels[s:]

    train_ds = Mydataset(train_imgs, train_labels, transform)
    test_ds = Mydataset(val_imgs, val_labels, transform)
    train_dl = data.DataLoader(train_ds, batch_size=2, shuffle=True)
    test_dl = data.DataLoader(test_ds, batch_size=2, shuffle=False)

    model = Vgg16_net()
    if torch.cuda.is_available():
        model.to('cuda')
    loss_fn = nn.L1Loss()
    optim = torch.optim.SGD(model.parameters(), lr=0.0001)
    epochs = 20

    # train
    for epoch in range(epochs):
        fit(epoch, model, loss_fn, optim, train_dl, test_dl)


    print("Finsh Training！")

