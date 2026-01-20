from ultralytics import YOLO
import os
import yaml
import shutil
import gc
import torch

# 判定標準：只看 mAP50 (x[2])
import ultralytics.utils.metrics
def custom_fitness(x):
    return x[2] # 只回傳 mAP@0.5，忽略 mAP@0.5-0.95
ultralytics.utils.metrics.fitness = custom_fitness
# =============================================================

def main():
    # 1. 路徑設定 (保持相對路徑)
    DATASET_DIR     = "Dataset"          
    TRAIN_REL_PATH  = "train/images"    
    VAL_REL_PATH    = "test/images"      
    TEST_REL_PATH   = "test/images"      
    CLASS_NAMES     = ['pig']
    
    # 2. 專案設定
    PROJECT_ROOT    = "runs/detect"
    PROJECT_NAME    = "train_pig_yolo11_x_1280_MAP50" 
    PRETRAINED_WGT  = 'yolo11x.pt'                 
    FINAL_WGT_NAME  = "111511167_v11x_best.pt"     

    # 3. 訓練參數
    EPOCHS_NUM      = 20      # 原始 40
    BATCH_SIZE      = 4      
    IMG_SIZE        = 1280   
    DEVICE_ID       = 0

    

    gc.collect()
    torch.cuda.empty_cache()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. YAML
    data_yaml_path = os.path.join(current_dir, "pig_dataset.yaml")
    
    data_dict = {
        'path': DATASET_DIR,        
        'train': TRAIN_REL_PATH,    
        'val': VAL_REL_PATH,
        'test': TEST_REL_PATH,
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_dict, f, sort_keys=False, allow_unicode=True)
    print(f"Dataset YAML 已更新: {data_yaml_path}")

    # 2. 強制清理舊訓練資料夾
    run_dir = os.path.join(current_dir, PROJECT_ROOT, PROJECT_NAME)
    if os.path.exists(run_dir):
        try:
            print(f"刪除舊資料夾: {run_dir}")
            shutil.rmtree(run_dir)
        except Exception as e:
            print(f"無法刪除: {e}")

    # 3. 載入 YOLO11-X 模型
    print(f"載入預訓練權重: {PRETRAINED_WGT}")
    model = YOLO(PRETRAINED_WGT)

    # 4. 訓練配置 
    train_config = {
        'data': data_yaml_path,
        'epochs': EPOCHS_NUM,
        'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE,
        'device': DEVICE_ID,
        'workers': 8,
        
        'project': PROJECT_ROOT,
        'name': PROJECT_NAME,
        'exist_ok': True,
        'save': True,
        'resume': False,

        #Parameter
        'optimizer': 'AdamW',   
        'lr0': 0.0001,         
        'lrf': 0.01,            
        'momentum': 0.9,        
        'weight_decay': 0.0001, 
        'warmup_epochs': 1.0,   #原始warmup_epoch : 3

        'mosaic': 1.0,          #原始mosaic : 0
        'mixup': 0.1,           #原始mixup : 0.0
        'close_mosaic': 1  ,    #原始close_mosaic:0
        'copy_paste': 0.0,      
        
        'scale': 0.5,           
        'fliplr': 0.5,          
        'translate': 0.1,       
        'degrees': 0.0,
        
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        
        'max_det': 300,
    }

    print("="*60)
    print(f"開始 YOLO11-X 訓練 (Best.pt 將鎖定 mAP50 最高者)")
    print("="*60)

    try:
        model.train(**train_config)
        print("\n訓練完成！")

    except torch.cuda.OutOfMemoryError:
        print("\nOOM Error: 顯存不足！")
        raise
    except Exception as e:
        print(f"\n訓練錯誤: {e}")
        raise

    # 5. 匯出結果
    best_weight_path = os.path.join(run_dir, "weights", "best.pt")
    target_weight_path = os.path.join(current_dir, FINAL_WGT_NAME)
    
    if os.path.exists(best_weight_path):
        shutil.copy(best_weight_path, target_weight_path)
        print(f"最佳模型已存: {target_weight_path}")
    else:
        print("未發現 best.pt")

if __name__ == '__main__':
    main()