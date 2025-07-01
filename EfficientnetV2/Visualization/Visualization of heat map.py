
from Evison import Display,show_network
from torchvision import models
from PIL import Image
import torch
from model import efficientnetv2_s as create_model
import os

device = 'cuda:0'
model = create_model(num_classes=36).to(device)
model_weight_path = "./weights/model-79.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

#print(show_network(model))

""" layer_list = ["blocks.2","blocks.6","blocks.10","blocks.15","blocks.20",
              "blocks.22","blocks.25","blocks.27","blocks.30", "blocks.35","blocks.39","head.project_conv"] """


layer = "head.project_conv.act"
file = "./Visualization/original"
image_list = os.listdir(file)
for img in image_list:
    path = os.path.join(file, img)
    image=Image.open(path).resize((384,384))
    name = os.path.basename(path)
    save_name = str(os.path.splitext(name)[0]) + str(layer)
    display = Display(model, layer, save_name=save_name, norm=((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),img_size=(384,384))
    display.save(image) 



 
