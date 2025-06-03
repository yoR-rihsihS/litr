# (LITR) License Transformer: Unified End-to-End License Plate Detection and Recognition

PyTorch implementation for License Transformer (MTech project at IISc Bangalore)

[Read Report](./Thesis.pdf) | [View Poster](./Poster.pdf) | [View Presentation: Detection Transformer](./Detection%20Transformers.pdf)

## Dataset

Download the CCPD dataset from [here](https://github.com/detectRecog/CCPD).

Make sure the project folder looks like this:
```
LITR/
├── CCPD2019/
│   ├── ccpd_base/
│   ├── ccpd_challenge/
│   └── ... (other directories and files)
├── codebase/
│   ├── backbone.py
│   ├── engine.py
│   └── ... (other python scripts defining the model)
├── configs/
│   ├── litr_r18.json
│   └── litr_r50.json
├── outputs/
│   └── ... (output images from visualizing predicitons)
├── samples/
│   └── ... (input images for visualizing predictions)
├── saved/
│   ├── litr_r18.pth
│   ├── litr_r50.pth
│   ├── litr_r50_finetuned.pth
│   └── ... (save directory during training and finetuning)
├── videos/
│   └── ... (input/output videos for visualizing predictions)
├── demo.py
├── evaluate.py
├── finetune.py
├── train.py
├── visualize.py
└── ... (other files from project)
```


## Training

- Run the following command to train LITR-R18:
```
python train.py --config "./configs/litr_r18.json"
```

- Run the following command to train LITR-R50:
```
python train.py --config "./configs/litr_r50.json"
```

- Run the following command to finetune model for practical predictions:
```
python finetune.py --config "{model config to finetune}" --checkpoint "{path to tranined model weights}"
```
Download finetuned LITR-R50 for predictions from [here](https://drive.google.com/file/d/14uQgXM3gO2Adr7XhfZ8EwXz7xwwt8tjm/view?usp=sharing).

## Performance

<table>
  <tr>
    <td>Model</td>
    <td>Input Shape</td>
    <td>Base</td>
    <td>DB</td>
    <td>FN</td>
    <td>Rotate</td>
    <td>Tilt</td>
    <td>Weather</td>
    <td>Challenge</td>
    <td>FPS</td>
    <td>Parameters (M)</td>
    <td>Model Weights</td>
  </tr>
  <tr>
    <td>LITR-R18</td>
    <td>640</td>
    <td>99.4</td>
    <td>86.7</td>
    <td>87.0</td>
    <td>93.5</td>
    <td>86.5</td>
    <td>97.9</td>
    <td>84.0</td>
    <td>50</td>
    <td>29.4</td>
    <td><a href="https://drive.google.com/file/d/1Qdz72qIb5dtp5aH4vB72K2rmF94Q6esj/view?usp=sharing">Downlaod</a></td>
  </tr>
  <tr>
    <td>LITR-R50</td>
    <td>640</td>
    <td>99.5</td>
    <td>89.0</td>
    <td>90.1</td>
    <td>93.6</td>
    <td>86.9</td>
    <td>98.5</td>
    <td>84.2</td>
    <td>41</td>
    <td>42.4</td>
    <td><a href="https://drive.google.com/file/d/1E5ZVF-eFkyGS3iyUHi2qlybGxFJV8LsZ/view?usp=sharing">Downlaod</a></td>
  </tr>
</table>

Comment: FPS is the number of images processed by model per second.

- Run the following command to evaluate LITR-R18 on test sets:
```
python evaluate.py --config "./configs/litr_r18.json" --model_path "./saved/litr_r18.pth"
```

- Run the following command to evaluate LITR-R50 on test sets:
```
python evaluate.py --config "./configs/litr_r50.json" --model_path "./saved/litr_r50.pth"
```


## Predictions

#### LITR-R50 outputs:

<table style="width: 100%;">
  <tr>
    <td><img src="./outputs/0.png" style="width: 100%;"/></td>
    <td><img src="./outputs/1.png" style="width: 100%;"/></td>
    <td><img src="./outputs/2.png" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="./outputs/3.png" style="width: 100%;"/></td>
    <td><img src="./outputs/4.png" style="width: 100%;"/></td>
    <td><img src="./outputs/5.png" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="./outputs/6.png" style="width: 100%;"/></td>
    <td><img src="./outputs/7.png" style="width: 100%;"/></td>
    <td><img src="./outputs/8.png" style="width: 100%;"/></td>
  </tr>
</table>

- Run the following command to get predictions over images:
```
python visualize.py
```

#### LITR-R50 output over video file:

![LITR-R50 output over video file](./videos/processed_01.gif)

Comment: [Visit](https://drive.google.com/file/d/1bm_mrDbdLNlUI8HJSZJU6Y9CFSptI9wb/view?usp=sharing) to view the full .mp4 output.

- Run the following command to get prediction over video file:
```
python demo.py
```