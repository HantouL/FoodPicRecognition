import random
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image  # 读取图片数据
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from model_utils.model import initialize_model  # 自定义初始化模型的方法
import matplotlib

matplotlib.use('TkAgg')  # 配置Matplotlib使用TkAgg后端


# 设置随机种子，确保结果的可复现性
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 关闭卷积的优化
    torch.backends.cudnn.deterministic = True  # 保证每次运行的结果一致
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 固定随机种子
seed_everything(0)

# 设置输入图像的大小
HW = 224

# 训练集数据变换（图像预处理）
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # 转换为PIL格式图像
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整为224x224大小
        transforms.RandomRotation(50),  # 随机旋转0到50度
        transforms.ToTensor()  # 转换为Tensor
    ]
)

# 验证集数据变换（简单的转换，避免过拟合）
val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor()
    ]
)


# 食物数据集类，继承自PyTorch的Dataset
class food_Dataset(Dataset):
    def __init__(self, path, mode="train"):
        self.mode = mode
        if mode == "semi":  # 半监督模式
            self.X = self.read_file(path)
        else:  # 标准的有标签数据模式
            self.X, self.Y = self.read_file(path)
            self.Y = torch.LongTensor(self.Y)  # 标签转为长整型

        # 根据模式选择不同的数据变换
        if mode == "train":
            self.transform = train_transform
        else:
            self.transform = val_transform

    # 读取数据文件
    def read_file(self, path):
        if self.mode == "semi":  # 半监督学习没有标签
            file_list = os.listdir(path)
            xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)  # 初始化图片数据数组
            for j, img_name in enumerate(file_list):
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path)
                img = img.resize((HW, HW))  # 调整图片大小
                xi[j, ...] = img
            print("读到了%d个数据" % len(xi))
            return xi
        else:  # 有标签数据集
            for i in tqdm(range(11)):  # 这里假设有11类食物
                file_dir = path + "/%02d" % i
                file_list = os.listdir(file_dir)
                xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
                yi = np.zeros(len(file_list), dtype=np.uint8)

                for j, img_name in enumerate(file_list):
                    img_path = os.path.join(file_dir, img_name)
                    img = Image.open(img_path)
                    img = img.resize((HW, HW))
                    xi[j, ...] = img
                    yi[j] = i

                if i == 0:
                    X = xi
                    Y = yi
                else:
                    X = np.concatenate((X, xi), axis=0)
                    Y = np.concatenate((Y, yi), axis=0)
            print("读到了%d个数据" % len(Y))
            return X, Y

    # 返回某个样本
    def __getitem__(self, item):
        if self.mode == "semi":
            return self.transform(self.X[item]), self.X[item]  # 半监督返回图片及原始图片
        else:
            return self.transform(self.X[item]), self.Y[item]  # 返回图片及标签

    def __len__(self):
        return len(self.X)


# 半监督学习的数据集类
class semiDataset(Dataset):
    def __init__(self, no_label_loder, model, device, thres=0.90):
        x, y = self.get_label(no_label_loder, model, device, thres)
        if x == []:
            self.flag = False  # 如果没有合格样本，标记为False
        else:
            self.flag = True
            self.X = np.array(x)
            self.Y = torch.LongTensor(y)
            self.transform = train_transform

    # 获取半监督数据的标签
    def get_label(self, no_label_loder, model, device, thres):
        model = model.to(device)
        pred_prob = []
        labels = []
        x = []
        y = []
        soft = nn.Softmax()
        with torch.no_grad():
            for bat_x, _ in no_label_loder:
                bat_x = bat_x.to(device)
                pred = model(bat_x)
                pred_soft = soft(pred)
                pred_max, pred_value = pred_soft.max(1)  # 获取概率最大的类别
                pred_prob.extend(pred_max.cpu().numpy().tolist())
                labels.extend(pred_value.cpu().numpy().tolist())

        # 选择置信度大于阈值的样本作为半监督样本
        for index, prob in enumerate(pred_prob):
            if prob > thres:
                x.append(no_label_loder.dataset[index][1])  # 调用到原始的getitem
                y.append(labels[index])
        return x, y

    def __getitem__(self, item):
        return self.transform(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.X)


# 获取半监督数据加载器
def get_semi_loader(no_label_loder, model, device, thres):
    semiset = semiDataset(no_label_loder, model, device, thres)
    if semiset.flag == False:
        return None
    else:
        semi_loader = DataLoader(semiset, batch_size=16, shuffle=False)
        return semi_loader


# 自定义模型
class myModel(nn.Module):
    def __init__(self, num_class):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)  # 卷积层，输出通道数64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()  # 激活函数
        self.pool1 = nn.MaxPool2d(2)  # 池化层

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(25088, 1000)  # 全连接层
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_class)  # 最终分类层

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)  # 拉平
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


# 训练和验证过程
def train_val(model, train_loader, val_loader, no_label_loader, device, epochs, optimizer, loss, thres, save_path):
    model = model.to(device)
    semi_loader = None
    plt_train_loss = []
    plt_val_loss = []
    plt_train_acc = []
    plt_val_acc = []

    max_acc = 0.0

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        semi_loss = 0.0
        semi_acc = 0.0

        start_time = time.time()

        model.train()  # 切换到训练模式
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_bat_loss = loss(pred, target)
            train_bat_loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # 梯度清零
            train_loss += train_bat_loss.cpu().item()
            train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

        plt_train_loss.append(train_loss / train_loader.__len__())
        plt_train_acc.append(train_acc / train_loader.dataset.__len__())

        # 进行半监督训练
        if semi_loader != None:
            for batch_x, batch_y in semi_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                semi_bat_loss = loss(pred, target)
                semi_bat_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                semi_loss += train_bat_loss.cpu().item()
                semi_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

        model.eval()  # 切换到评估模式
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target)
                val_loss += val_bat_loss.cpu().item()
                val_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

        plt_val_loss.append(val_loss / val_loader.dataset.__len__())
        plt_val_acc.append(val_acc / val_loader.dataset.__len__())

        if epoch % 3 == 0 and plt_val_acc[-1] > 0.01:
            semi_loader = get_semi_loader(no_label_loader, model, device, thres)

        # 保存最佳模型
        if val_acc > max_acc:
            torch.save(model, save_path)
            max_acc = val_acc

        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %.6f | valLoss: %.6f Trainacc : %.6f | valacc: %.6f' % \
              (epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1], plt_train_acc[-1],
               plt_val_acc[-1]))

    # 绘制损失和准确率曲线
    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend(["train", "val"])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title("acc")
    plt.legend(["train", "val"])
    plt.show()


# 设置路径
train_path = r"E:\DLDataset\Food\food-11_sample\training\labeled"
val_path = r"E:\DLDataset\Food\food-11_sample\validation"
no_label_path = r"E:\DLDataset\Food\food-11_sample\training\unlabeled\00"

# 创建数据集
train_set = food_Dataset(train_path, "train")
val_set = food_Dataset(val_path, "val")
no_label_set = food_Dataset(no_label_path, "semi")

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
no_label_loader = DataLoader(no_label_set, batch_size=16, shuffle=False)

# 初始化模型（使用VGG作为示例）
# model = myModel(11)
model, _ = initialize_model("vgg", 11, use_pretrained=True)

# 设置优化器和损失函数
lr = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 使用AdamW优化器

# 训练过程
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "model_save/best_model.pth"
epochs = 30
thres = 0.3

train_val(model, train_loader, val_loader, no_label_loader, device, epochs, optimizer, loss, thres, save_path)
