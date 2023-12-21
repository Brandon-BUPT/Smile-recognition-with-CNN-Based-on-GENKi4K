import glob
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*10*10, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # print(x.size())
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def fit(epoch, model,loss_fn, optim, train_dl, test_dl):
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


if __name__ == "__main__" :
    imagePaths = glob.glob(".\\Data\\gender\\files\\*.jpg")
    labelPaths = read_first_column(".\\Data\\gender\\labels.txt")
    # print(labelPaths)
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    index = np.random.permutation(len(imagePaths))
    all_imgs_path = np.array(imagePaths)[index]
    all_labels = np.array(labelPaths)[index]

    s = int(len(all_imgs_path) * 0.8)
    train_imgs = all_imgs_path[:s]
    train_labels = all_labels[:s]
    val_imgs = all_imgs_path[s:]
    val_labels = all_labels[s:]

    train_ds = Mydataset(train_imgs, train_labels, transform)
    test_ds = Mydataset(val_imgs, val_labels, transform)
    train_dl = data.DataLoader(train_ds, batch_size=2, shuffle=True)
    test_dl = data.DataLoader(test_ds, batch_size=2, shuffle=False)

    model = Net()
    if torch.cuda.is_available():
        model.to('cuda')
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30

    # train
    for epoch in range(epochs):
        fit(epoch, model, loss_fn, optim, train_dl, test_dl)


    print("Finsh Training！")

