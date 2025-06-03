import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import torch

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

from codebase import LiTr, LiTrPostProcessor
DEVICE = "cuda"

cfg = json.load(open("./configs/litr_r50.json", 'r'))

model = LiTr(
    num_classes = cfg['num_classes'],
    backbone_model = cfg['backbone_model'],
    hidden_dim = cfg['hidden_dim'], 
    nhead = cfg['nhead'], 
    ffn_dim = cfg['ffn_dim'], 
    num_encoder_layers = cfg['num_encoder_layers'], 
    eval_spatial_size = cfg['eval_spatial_size'],
    aux_loss = cfg['aux_loss'],
    num_queries = cfg['num_queries'],
    num_decoder_points = cfg['num_decoder_points'],
    num_denoising = cfg['num_denoising'],
    num_decoder_layers = cfg['num_decoder_layers'],
    dropout = cfg['dropout'],
)
model.to(DEVICE)
checkpoint = torch.load(f"./saved/{cfg['model_name']}_finetuned.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

postprocessor = LiTrPostProcessor(num_classes=cfg['num_classes'], num_queries = cfg['num_queries']) 

files = []
for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'Noto' in font or 'CJK' in font or 'PingFang' in font or 'Songti' in font:
        print(font)
        files.append(font)
assert len(files) >= 2, "No fonts found for Chinese characters."


def plot_result(image, outputs, name):
    boxes, scores, labels = outputs
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy().round(3)
    h, w = image.shape[:2]

    # Convert cxcywh → xyxy (absolute pixel coordinates)
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0 = (cx - bw / 2) * w
    y0 = (cy - bh / 2) * h
    x1 = (cx + bw / 2) * w
    y1 = (cy + bh / 2) * h

    chinese_font = font_manager.FontProperties(fname=files[1], size=10)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i in range(boxes.shape[0]):
        rect = patches.Rectangle(
            (x0[i], y0[i]),
            x1[i] - x0[i],
            y1[i] - y0[i],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)

        label_text = provinces[labels[i, 0]] + alphabets[labels[i, 1]]
        for j in range(2, 7):
            label_text += ads[labels[i, j]]

        ax.text(
            x0[i], y0[i] - 12,
            label_text,
            fontproperties=chinese_font,
            fontsize=8,
            color='white',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1.5)
        )

        ax.text(
            x1[i], y1[i] + 24,
            str(scores[i, 0]),
            fontproperties=chinese_font,
            fontsize=8,
            color='white',
            bbox=dict(facecolor='blue', alpha=0.5, edgecolor='none', pad=1.5)
        )

    ax.axis('off')
    plt.savefig('./outputs/'+str(name)+'.png', dpi=300, bbox_inches='tight')
    plt.close()

mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

file_names = []
for file_name in os.listdir("./samples/"):
    file_names.append(file_name)

for i, file_name in enumerate(file_names):
    img = cv2.imdecode(np.fromfile("./samples/"+file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(img, (640, 640))
    input_image = input_image.astype(np.float32) / 255.0 # Scale to [0,1]
    input_image = np.transpose(input_image, (2, 0, 1))   # From HWC to CHW 
    input_image = np.ascontiguousarray(input_image)    
    input_image = (input_image - mean) / std # Normalize
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dim
    input_image = input_image.astype(np.float32)
    input_image = torch.from_numpy(input_image).to(DEVICE)

    outputs = model(input_image)
    outputs = postprocessor(outputs, top_k=100, score_thresh=0.5)
    boxes, scores, labels = outputs[0]["boxes"], outputs[0]["scores"], outputs[0]["labels"]

    plot_result(img, (boxes, scores, labels), i)
    print(f"Processesed {i+1}/{len(file_names)} images", end=' \r')