import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import numpy as np
from bs4 import BeautifulSoup


class MyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.label=list(sorted(os.listdir(os.path.join(root, "annotations"))))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "images", self.images[idx])
        label_path = os.path.join(self.root, "annotations", self.label[idx])

        x = cv2.imread(image_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.array(x, dtype=np.float32) / 255
        x = torch.from_numpy(x)
        x = torch.permute(x, (2, 0, 1))
        y = {}
        with open(label_path) as f:
            data = f.read()
            soup = BeautifulSoup(data, "xml")
            objs = soup.find_all("object")
            y["boxes"] = []
            y["labels"] = []
            y["image_id"] = torch.tensor([idx])
            for obj in objs:
                x1 = int(obj.find("xmin").text)
                y1 = int(obj.find("ymin").text)
                x2 = int(obj.find("xmax").text)
                y2 = int(obj.find("ymax").text)
                y["boxes"].append([x1, y1, x2, y2])
                label = 0
                if obj.find("name").text == "obj1": label = 1
                if obj.find("name").text == "obj2": label = 2
                if obj.find("name").text == "obj3": label = 3
                if obj.find("name").text == "obj4": label = 4
                if obj.find("name").text == "obj5": label = 5
                y["labels"].append(label)
            y["boxes"] = torch.as_tensor(y["boxes"], dtype=torch.float32)  # data type
            y["labels"] = torch.as_tensor(y["labels"], dtype=torch.int64)
        return x, y


def collate_fn(batch): return tuple(zip(*batch))


device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 6
net = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = net.roi_heads.box_predictor.cls_score.in_features
net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
if os.path.exists("pmodel.pth"):
    net.load_state_dict(torch.load("pmodel.pth"))
net = net.to(device)

batch_size = 1
train_data = MyDataset("C:\\Users\\user\\Desktop\\tool_detecte")
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True
)

writer = SummaryWriter("runs")
params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
)

start_from = 1
num_epochs = 10
e = 0
while True:
    net.train()
    running_loss = 0

    for batch, (x, y) in enumerate(train_loader, 1):
        x = [i.to(device) for i in x]
        y = [{k: v.to(device) for k, v in i.items()} for i in y]

        loss_dict = net(x, y)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        print("Epoch %d(%d/%d): train_loss: %.4f" % (e, batch, len(train_loader), running_loss / batch))
    running_loss = running_loss / len(train_loader)
    writer.add_scalars("loss", {"train": running_loss}, e)
    writer.flush()
    print("Epoch %d: train_loss: %.4f" % (e, running_loss))

    torch.save(net.state_dict(), "pmodel.pth")
    e += 1