import os
import yaml
import tempfile
from ultralytics import YOLO

def dict_to_temp_yaml(cfg: dict):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
    with open(tmp.name, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    return tmp.name

# 1. Dataset設定
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(current_dir, "Dataset")
DATA_ROOT = DATA_ROOT.replace("\\", "/")

data_dict = {
    'path': DATA_ROOT,
    'train': "train/images",
    'val':   "test/images",
    'test':  "test/images",
    'nc': 1,
    'names': ['pig']
}

data_yaml_path = dict_to_temp_yaml(data_dict)

# 2. 載入 Model
model_path = os.path.join(current_dir, "111511167.pt") 
if not os.path.exists(model_path):
    model_path = os.path.join(current_dir, "runs/detect/train_pig_detection/weights/best.pt")

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# 3. Test Set metrics 計算
print("Starting Evaluation on Test Set...")
metrics = model.val(
    data=data_yaml_path, 
    split="test",
    imgsz=1280,
    batch=32,
    conf=0.001,
    iou=0.65,          #原始=0.6
    verbose=True,
    plots=True,
    device=0
)

ap50 = metrics.box.map50
mAP = metrics.box.map
precision = metrics.box.mp
recall = metrics.box.mr
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

print("\n===== Test Results =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"AP@50:     {ap50:.4f}")
print(f"mAP50-95:  {mAP:.4f}")

# 4. Inference & Visualization
print("\nRunning Inference...")
test_img_dir = os.path.join(DATA_ROOT, "test/images")
output_dir = os.path.join(current_dir, "inference_results")

results = model.predict(
    source=test_img_dir,
    conf=0.25,
    iou=0.55,
    save=True,
    project=output_dir,
    name='visualizations',
    exist_ok=True
)

print(f"Done. Visualized images saved in {output_dir}/visualizations/")