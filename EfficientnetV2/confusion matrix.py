import json
import os
from model import efficientnetv2_s as create_model
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import torch
from torchvision import transforms
from model import efficientnetv2_s as create_model
from utils import read_split_data
from my_dataset import MyDataSet
from tqdm import tqdm
import csv

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    def summary(self):
        sum_tp = 0
        for i in range(self.num_classes):
            sum_tp += self.matrix[i,i]
        acc = sum_tp / np.sum(self.matrix)
        print("the model accuraacy is", acc)

        """ #precision,recall, specifcity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i,i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 4)
            Recall = round(TP / (TP + FN), 4)
            Specificity = round(TN / (TN + FP), 4)
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table) """

        table = PrettyTable()
        table.field_names = ["label", "Precision", "Recall", "Specificity"]

        # 将 PrettyTable 数据写入 CSV 文件
        csv_filename = "metrics.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["label", "Precision", "Recall", "Specificity"]) # 写入表头
            for i in range(self.num_classes):
                TP = self.matrix[i,i]
                FP = np.sum(self.matrix[i, :]) - TP
                FN = np.sum(self.matrix[:, i]) - TP
                TN = np.sum(self.matrix) - TP - FP - FN
                Precision = round(TP / (TP + FP), 4)
                Recall = round(TP / (TP + FN), 4)
                Specificity = round(TN / (TN + FP), 4)
                table.add_row([self.labels[i], Precision, Recall, Specificity])
                csv_writer.writerow([self.labels[i], Precision, Recall, Specificity])
        print(table)
        print(f"CSV file '{csv_filename}' has been created.")

    def plot(self):
        matrix = self.matrix
        #print(matrix)
        plt.imshow(matrix, cmap= plt.cm.Blues)
        
        #设置X轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)

        #设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)

        #显示colorbar
        plt.colorbar()
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Confusion matrix")

        #在图中标注数量/概率信息
        thresh = matrix.max() / 2 #设置一个阈值来决定图中数字的颜色
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                #注意这里的是matrix[y,x],不是[x,y]
                info = int(matrix[y, x])
                plt.text(x, y, info, 
                         verticalalignment="center",
                         horizontalalignment = "center",
                         color = "white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                         transforms.CenterCrop(img_size[num_model][1]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    _, _, val_images_path, val_images_label = read_split_data("./data/jiance_fenlei")
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)
    
    
    net = create_model(num_classes=9)
    model_weight_path = "./weights/model-79.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)
    
    try:
        json_file = open("./class_indices.json", "r")
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=9, labels=labels)
    net.eval()

    with torch.no_grad():
        for val_data in tqdm(val_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.cpu().detach().numpy(), val_labels.cpu().detach().numpy())
    confusion.plot() #类太多啦，生成的混淆矩阵会乱码
    confusion.summary()



    
      