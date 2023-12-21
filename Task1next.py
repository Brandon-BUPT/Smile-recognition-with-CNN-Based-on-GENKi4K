import glob
import numpy as np
from PIL import Image
import torch
import os
from torch import nn
from torch.utils import data
from torchvision import transforms


def read_first_column(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        first_column = [int(line.strip().split(' ')[0]) for line in lines]
    return first_column


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

            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def fit(epoch, model,loss_fn, optim, train_dl,test_dl, save_dir):
    correct = 0
    total = 0
    model.train()
    for x, y in train_dl:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y.long())
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
    train_acc = correct / total

    test_correct = 0
    test_total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
    test_acc = test_correct / test_total
    print('epoch: ', epoch, 'train_accuracy:', round(train_acc, 3), 'test_accuracy:', round(test_acc, 3))



def test_model(model, test_dl):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f'Test Accuracy: {acc * 100:.2f}%')

if __name__ == "__main__":
    imagePaths = glob.glob("Data/genki4k/files/*.jpg")
    labelPaths = read_first_column("/lab/Mini Project/Data/genki4k/labels.txt")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    index = np.random.permutation(len(imagePaths))
    all_imgs_path = np.array(imagePaths)[index]
    all_labels = np.array(labelPaths)[index]

    # 保存
    save_dir = 'save'
    os.makedirs(save_dir, exist_ok=True)

    # 定义训练集、验证集和测试集的大小
    train_size = int(len(all_imgs_path) * 0.7)
    val_size = int(len(all_imgs_path) * 0.15)

    # 划分训练集、验证集和测试集
    train_imgs = all_imgs_path[:train_size]
    train_labels = all_labels[:train_size]
    val_imgs = all_imgs_path[train_size:train_size+val_size]
    val_labels = all_labels[train_size:train_size+val_size]
    test_imgs = all_imgs_path[train_size+val_size:]
    test_labels = all_labels[train_size+val_size:]

    # 创建数据集
    train_ds = Mydataset(train_imgs, train_labels, transform)
    val_ds = Mydataset(val_imgs, val_labels, transform)
    test_ds = Mydataset(test_imgs, test_labels, transform)

    # 创建数据加载器
    train_dl = data.DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = data.DataLoader(val_ds, batch_size=4, shuffle=False)
    test_dl = data.DataLoader(test_ds, batch_size=4, shuffle=False)

    model = Vgg16_net()
    if torch.cuda.is_available():
        model.to('cuda')
    # 加载保存的模型权重
    model_path = os.path.join('save', 'model_epoch_9.pth')
    model.load_state_dict(torch.load(model_path))

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 10

    # 训练模型
    for epoch in range(epochs):
            fit(epoch, model, loss_fn, optim, train_dl, val_dl, save_dir)

    # 保存模型
    save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    print("Finish Training!")
    # 在测试集上测试模型
    test_model(model, test_dl)