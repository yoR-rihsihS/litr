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
checkpoint = torch.load(f"./saved_new/{cfg['model_name']}_25.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

postprocessor = LiTrPostProcessor(num_classes=cfg['num_classes'], num_queries = cfg['num_queries']) 

files = []
for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'Noto' in font or 'CJK' in font or 'PingFang' in font or 'Songti' in font:
        print(font)
        files.append(font)
assert len(files) >= 2, "No fonts found for Chinese characters."

def process_frame(image, outputs):
    """
    Process a single frame to add bounding boxes and license plate text.
    Returns the annotated frame as a NumPy array (RGB).
    """
    boxes, scores, labels = outputs
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy().round(3)

    h, w = image.shape[:2]
    aspect_ratio = w / h

    # Convert cxcywh → xyxy (absolute pixel coordinates)
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0 = (cx - bw / 2) * w
    y0 = (cy - bh / 2) * h
    x1 = (cx + bw / 2) * w
    y1 = (cy + bh / 2) * h

    # Set Chinese-compatible font
    chinese_font = font_manager.FontProperties(fname=files[1], size=10)

    # Create figure with adjusted size
    fig, ax = plt.subplots(1, figsize=(16, 16 / aspect_ratio))
    canvas = FigureCanvas(fig)  # Attach Agg canvas
    ax.imshow(image)
    for i in range(boxes.shape[0]):
        rect = patches.Rectangle(
            (x0[i], y0[i]),
            x1[i] - x0[i],
            y1[i] - y0[i],
            linewidth=1,
            edgecolor='r',
            facecolor='none',
            antialiased=True
        )
        ax.add_patch(rect)

        # Construct license plate text
        label_text = provinces[labels[i, 0]] + alphabets[labels[i, 1]]
        for j in range(2, 7):
            label_text += ads[labels[i, j]]

        ax.text(
            x0[i], y0[i] - 12,
            label_text,
            fontproperties=chinese_font,
            color='white',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1.5)
        )

        ax.text(
            x1[i], y1[i] + 24,
            str(scores[i]),
            fontproperties=chinese_font,
            color='white',
            bbox=dict(facecolor='blue', alpha=0.5, edgecolor='none', pad=1.5)
        )

    ax.axis('off')
    plt.tight_layout()

    # Convert Matplotlib figure to NumPy array
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]
    plt.close(fig)

    # Resize to match input frame dimensions
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img

mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

def inference(image):
    """
    Placeholder for your license plate detection model.
    Replace with your actual model inference code.
    Returns (boxes, scores, labels) as expected by process_frame.
    """
    input_image = cv2.resize(image, (640, 640))
    input_image = input_image.astype(np.float32) / 255.0 # Scale to [0, 1]
    input_image = np.transpose(input_image, (2, 0, 1))   # From HWC to CHW 
    input_image = np.ascontiguousarray(input_image)    
    input_image = (input_image - mean) / std # Normalize
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dim
    input_image = input_image.astype(np.float32)
    input_image = torch.from_numpy(input_image).to(DEVICE)

    outputs = model(input_image)
    outputs = postprocessor(outputs, top_k=100, score_thresh=0.33)
    boxes, scores, labels = outputs[0]["boxes"], outputs[0]["scores"], outputs[0]["labels"]

    return boxes, scores, labels

def process_video(input_path, output_path, inference_func=inference):
    """
    Process a video by applying inference and annotations to each frame.
    Saves the result as a new video.
    Args:
        - input_path (str): Path to input video.
        - output_path (str): Path to save output video.
        - inference_func (callable): Function that takes a frame and returns (boxes, scores, labels).
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not initialize video writer: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference (replace dummy_inference with your model)
        outputs = inference_func(frame_rgb)

        # Process frame to add annotations
        processed_frame = process_frame(frame_rgb, outputs)

        # Convert RGB back to BGR for OpenCV
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Write frame to output video
        out.write(processed_frame_bgr)

        frame_count += 1
        if frame_count % 100 == 0 or frame_count == total_frames:
            print(f"Processed frame {frame_count}/{total_frames}", end='\r')

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVideo saved to {output_path}")

input_video = "./videos/01.mp4"
output_video = "./videos/processed_01.mp4"

try:
    process_video(input_video, output_video, inference_func=inference)
except Exception as e:
    print(f"Error: {e}")