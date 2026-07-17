!pip install -q ultralytics
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.1/42.1 kB 2.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 38.3 MB/s eta 0:00:00

from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
     
Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.

from zipfile import ZipFile

with ZipFile("/content/food_img_data_resized (1).zip","r") as zip_ref:
    zip_ref.extractall("/content")
     

model = YOLO("yolov8n-cls.pt")
     
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-cls.pt to 'yolov8n-cls.pt': 100% ━━━━━━━━━━━━ 5.3MB 98.4MB/s 0.1s

model.train(
    data="/content/food_img_data_resized",   # Dataset folder
    epochs=30,
    imgsz=224,
    batch=32,
    device=0
)
     
Ultralytics 8.4.98 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/food_img_data_resized, degrees=0.0, deterministic=True, device=0, dfl=1.5, dis=6.0, distill_model=None, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=30, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=224, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n-cls.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=train, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/runs/classify/train, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=classify, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
WARNING ⚠️ Dataset 'split=train' not found at /content/food_img_data_resized/train
Found 1664 images in subdirectories. Attempting to split...
Splitting /content/food_img_data_resized (15 classes, 1664 images) into 80% train, 20% val...
Split complete in /content/food_img_data_resized_split ✅
train: /content/food_img_data_resized_split/train... found 1328 images in 15 classes ✅ 
val: /content/food_img_data_resized_split/val... found 336 images in 15 classes ✅ 
test: None...
Overriding model.yaml nc=1000 with nc=15

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    349455  ultralytics.nn.modules.head.Classify         [256, 15]                     
YOLOv8n-cls summary: 56 layers, 1,457,503 parameters, 1,457,503 gradients, 3.4 GFLOPs
Transferred 156/158 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 96.8MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 453.6±252.7 MB/s, size: 12.4 KB)
train: Scanning /content/food_img_data_resized_split/train... 1328 images, 0 corrupt: 100% ━━━━━━━━━━━━ 1328/1328 3.8Kit/s 0.3s
train: New cache created: /content/food_img_data_resized_split/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 249.3±162.9 MB/s, size: 13.0 KB)
val: Scanning /content/food_img_data_resized_split/val... 336 images, 0 corrupt: 100% ━━━━━━━━━━━━ 336/336 2.4Kit/s 0.1s
val: New cache created: /content/food_img_data_resized_split/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.000526, momentum=0.9) with parameter groups 26 weight(decay=0.0), 27 weight(decay=0.0005), 27 bias(decay=0.0)
Image sizes 224 train, 224 val
Using 2 dataloader workers
Logging results to /content/runs/classify/train
Starting training for 30 epochs...

      Epoch    GPU_mem       loss  Instances       Size
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 18.2MB/s 0.0s
       1/30     0.416G      2.572         16        224: 100% ━━━━━━━━━━━━ 42/42 4.4it/s 9.6s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 7.6it/s 0.8s
                   all      0.449      0.845

      Epoch    GPU_mem       loss  Instances       Size
       2/30     0.516G      1.662         16        224: 100% ━━━━━━━━━━━━ 42/42 6.8it/s 6.2s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 21.1it/s 0.3s
                   all      0.762      0.979

      Epoch    GPU_mem       loss  Instances       Size
       3/30     0.516G     0.8376         16        224: 100% ━━━━━━━━━━━━ 42/42 5.2it/s 8.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 33.2it/s 0.2s
                   all      0.836      0.982

      Epoch    GPU_mem       loss  Instances       Size
       4/30     0.516G     0.4985         16        224: 100% ━━━━━━━━━━━━ 42/42 5.3it/s 8.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 19.2it/s 0.3s
                   all      0.899      0.988

      Epoch    GPU_mem       loss  Instances       Size
       5/30     0.516G     0.4012         16        224: 100% ━━━━━━━━━━━━ 42/42 5.8it/s 7.3s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 27.6it/s 0.2s
                   all      0.896      0.988

      Epoch    GPU_mem       loss  Instances       Size
       6/30     0.516G     0.3105         16        224: 100% ━━━━━━━━━━━━ 42/42 5.5it/s 7.7s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 17.9it/s 0.3s
                   all      0.893      0.991

      Epoch    GPU_mem       loss  Instances       Size
       7/30     0.516G     0.2827         16        224: 100% ━━━━━━━━━━━━ 42/42 3.8it/s 11.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 19.1it/s 0.3s
                   all      0.911      0.991

      Epoch    GPU_mem       loss  Instances       Size
       8/30     0.516G     0.2394         16        224: 100% ━━━━━━━━━━━━ 42/42 6.6it/s 6.3s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 30.6it/s 0.2s
                   all      0.935      0.991

      Epoch    GPU_mem       loss  Instances       Size
       9/30     0.516G      0.201         16        224: 100% ━━━━━━━━━━━━ 42/42 5.3it/s 8.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 30.9it/s 0.2s
                   all      0.946      0.991

      Epoch    GPU_mem       loss  Instances       Size
      10/30     0.516G      0.186         16        224: 100% ━━━━━━━━━━━━ 42/42 6.5it/s 6.4s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 26.2it/s 0.2s
                   all      0.943      0.994

      Epoch    GPU_mem       loss  Instances       Size
      11/30     0.516G     0.1451         16        224: 100% ━━━━━━━━━━━━ 42/42 4.6it/s 9.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 29.2it/s 0.2s
                   all      0.943      0.988

      Epoch    GPU_mem       loss  Instances       Size
      12/30     0.516G     0.1396         16        224: 100% ━━━━━━━━━━━━ 42/42 5.2it/s 8.2s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 36.1it/s 0.2s
                   all       0.94      0.988

      Epoch    GPU_mem       loss  Instances       Size
      13/30     0.516G     0.1598         16        224: 100% ━━━━━━━━━━━━ 42/42 6.5it/s 6.5s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 26.7it/s 0.2s
                   all      0.938      0.991

      Epoch    GPU_mem       loss  Instances       Size
      14/30     0.516G     0.1252         16        224: 100% ━━━━━━━━━━━━ 42/42 5.3it/s 7.9s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 29.3it/s 0.2s
                   all      0.955      0.991

      Epoch    GPU_mem       loss  Instances       Size
      15/30     0.516G     0.1082         16        224: 100% ━━━━━━━━━━━━ 42/42 5.7it/s 7.4s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 14.0it/s 0.4s
                   all      0.952      0.991

      Epoch    GPU_mem       loss  Instances       Size
      16/30     0.516G    0.09661         16        224: 100% ━━━━━━━━━━━━ 42/42 5.9it/s 7.1s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 25.2it/s 0.2s
                   all      0.952      0.991

      Epoch    GPU_mem       loss  Instances       Size
      17/30     0.516G      0.103         16        224: 100% ━━━━━━━━━━━━ 42/42 5.1it/s 8.3s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 23.2it/s 0.3s
                   all      0.955      0.994

      Epoch    GPU_mem       loss  Instances       Size
      18/30     0.516G     0.1074         16        224: 100% ━━━━━━━━━━━━ 42/42 6.5it/s 6.5s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 27.8it/s 0.2s
                   all      0.964      0.994

      Epoch    GPU_mem       loss  Instances       Size
      19/30     0.516G    0.09935         16        224: 100% ━━━━━━━━━━━━ 42/42 4.8it/s 8.7s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 29.0it/s 0.2s
                   all      0.961      0.994

      Epoch    GPU_mem       loss  Instances       Size
      20/30     0.516G    0.09821         16        224: 100% ━━━━━━━━━━━━ 42/42 5.2it/s 8.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 23.0it/s 0.3s
                   all      0.949      0.991

      Epoch    GPU_mem       loss  Instances       Size
      21/30     0.516G     0.1221         16        224: 100% ━━━━━━━━━━━━ 42/42 5.5it/s 7.6s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 25.3it/s 0.2s
                   all      0.946      0.991

      Epoch    GPU_mem       loss  Instances       Size
      22/30     0.516G    0.08177         16        224: 100% ━━━━━━━━━━━━ 42/42 4.3it/s 9.7s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 22.3it/s 0.3s
                   all      0.943      0.991

      Epoch    GPU_mem       loss  Instances       Size
      23/30     0.516G    0.07962         16        224: 100% ━━━━━━━━━━━━ 42/42 5.3it/s 7.9s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 19.1it/s 0.3s
                   all      0.952      0.991

      Epoch    GPU_mem       loss  Instances       Size
      24/30     0.516G    0.07511         16        224: 100% ━━━━━━━━━━━━ 42/42 6.7it/s 6.3s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 26.6it/s 0.2s
                   all      0.961      0.991

      Epoch    GPU_mem       loss  Instances       Size
      25/30     0.516G    0.08875         16        224: 100% ━━━━━━━━━━━━ 42/42 5.2it/s 8.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 23.7it/s 0.3s
                   all      0.964      0.991

      Epoch    GPU_mem       loss  Instances       Size
      26/30     0.516G     0.1078         16        224: 100% ━━━━━━━━━━━━ 42/42 5.5it/s 7.7s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 20.9it/s 0.3s
                   all      0.967      0.991

      Epoch    GPU_mem       loss  Instances       Size
      27/30     0.516G    0.07812         16        224: 100% ━━━━━━━━━━━━ 42/42 5.8it/s 7.2s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 20.1it/s 0.3s
                   all       0.97      0.991

      Epoch    GPU_mem       loss  Instances       Size
      28/30     0.516G       0.07         16        224: 100% ━━━━━━━━━━━━ 42/42 5.1it/s 8.2s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 31.6it/s 0.2s
                   all      0.961      0.997

      Epoch    GPU_mem       loss  Instances       Size
      29/30     0.516G    0.07173         16        224: 100% ━━━━━━━━━━━━ 42/42 6.1it/s 6.8s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 34.5it/s 0.2s
                   all      0.964      0.991

      Epoch    GPU_mem       loss  Instances       Size
      30/30     0.516G    0.07407         16        224: 100% ━━━━━━━━━━━━ 42/42 4.3it/s 9.8s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 18.7it/s 0.3s
                   all      0.967      0.991

30 epochs completed in 0.075 hours.
Optimizer stripped from /content/runs/classify/train/weights/last.pt, 3.0MB
Optimizer stripped from /content/runs/classify/train/weights/best.pt, 3.0MB

Validating /content/runs/classify/train/weights/best.pt...
Ultralytics 8.4.98 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLOv8n-cls summary (fused): 30 layers, 1,454,095 parameters, 0 gradients, 3.3 GFLOPs
WARNING ⚠️ Dataset 'split=train' not found at /content/food_img_data_resized/train
Found 1664 images in subdirectories. Attempting to split...
Splitting /content/food_img_data_resized (15 classes, 1664 images) into 80% train, 20% val...
Split complete in /content/food_img_data_resized_split ✅
train: /content/food_img_data_resized_split/train... found 1602 images in 15 classes ✅ 
val: /content/food_img_data_resized_split/val... found 610 images in 15 classes ✅ 
test: None...
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 6/6 14.7it/s 0.4s
                   all       0.97      0.991
Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.0ms postprocess per image
Results saved to /content/runs/classify/train
ultralytics.utils.metrics.ClassifyMetrics object with attributes:

confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x78188c6364b0>
curves: []
curves_results: []
fitness: 0.9806547462940216
keys: ['metrics/accuracy_top1', 'metrics/accuracy_top5']
results_dict: {'metrics/accuracy_top1': 0.9702380895614624, 'metrics/accuracy_top5': 0.9910714030265808, 'fitness': 0.9806547462940216}
save_dir: PosixPath('/content/runs/classify/train')
speed: {'preprocess': 0.1418464642854577, 'inference': 0.7050351160716992, 'loss': 0.00020338988059891215, 'postprocess': 0.000295389881105599}
top1: 0.9702380895614624
top5: 0.9910714030265808

import glob

best_model = glob.glob("/content/runs/classify/**/weights/best.pt", recursive=True)

print(best_model)
     
['/content/runs/classify/train/weights/best.pt']

model = YOLO(best_model[0])
     

nutrition = pd.read_excel("/content/Nutrition_Table_15_Food_Classes.xlsx")

nutrition.head()
     
Food Name	Serving Size (g)	Calories (kcal)	Protein (g)	Carbohydrates (g)	Fat (g)	Fiber (g)	Sugar (g)	Sodium (mg)
0	Biryani	350	720	32	78	28	3	3	1200
1	Pizza	250	620	22	70	24	4	6	950
2	Vegetable Salad	200	180	5	18	8	6	7	220
3	Fruits Salad	250	150	2	38	1	5	30	10
4	Burger	250	580	26	45	30	4	8	950

from google.colab import files

uploaded = files.upload()

     
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving food_img_data_resized (1).zip to food_img_data_resized (1) (3).zip
To ensure the model is defined, run the following cell. This cell loads the best-trained weights into a YOLO model instance.


model = YOLO(best_model[0])
     

!pip install -q ultralytics gradio openpyxl

import gradio as gr
import pandas as pd
from ultralytics import YOLO

# Load Model
model = YOLO("/content/runs/classify/train/weights/best.pt")

# Load Nutrition Table
nutrition = pd.read_excel("/content/Nutrition_Table_15_Food_Classes.xlsx")

# Check column names
print("Columns in Excel:")
print(nutrition.columns.tolist())

# Change this if your first column has a different name
food_column = nutrition.columns[0]

def predict_food(image):
    try:
        # Predict
        results = model.predict(image, verbose=False)

        predicted = results[0].names[results[0].probs.top1]
        confidence = float(results[0].probs.top1conf) * 100

        print("Prediction:", predicted)

        # Find nutrition row
        food = nutrition[
            nutrition[food_column].astype(str).str.lower() == predicted.lower()
        ]

        if not food.empty:

            row = food.iloc[0]

            nutrition_text = ""

            for col in nutrition.columns:
                nutrition_text += f"{col}: {row[col]}\n"

        else:

            nutrition_text = "Nutrition information not found."

        return (
            predicted,
            f"{confidence:.2f} %",
            nutrition_text
        )

    except Exception as e:
        print(e)
        return (
            "Error",
            "Error",
            str(e)
        )

demo = gr.Interface(
    fn=predict_food,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Predicted Food"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Nutrition Information")
    ],
    title="Food Detection using YOLO",
    description="Upload a food image and click Submit."
)

demo.launch(debug=True)
     
Columns in Excel:
['Food Name', 'Serving Size (g)', 'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)', 'Sugar (g)', 'Sodium (mg)']
It looks like you are running Gradio on a hosted Jupyter notebook, which requires `share=True`. Automatically setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).

Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().
* Running on public URL: https://e5b9cccb599f441c8a.gradio.live

This share link is temporary and will last for up to 1 week (best effort). For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
Prediction: noodles
Prediction: Pavbhaji
Prediction: pizza
Prediction: pasta
Prediction: sandwiches
Prediction: sushi
Prediction: dosa
