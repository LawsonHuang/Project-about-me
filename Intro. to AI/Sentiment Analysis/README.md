# HW3: Sentiment Analysis with Deep Learning
本專案使用 Microsoft DeBERTa V3 Large 模型進行情感分析。針對作業需求，代碼支援單一模型訓練以及在高性能 GPU (如 NVIDIA A100) 上進行多組超參數平行訓練。

## 成果  
我使用 DeBERTa-v3-large 模型，在總參數量(434.02M)必須低於500M下，達成 Accuracy=0.8535，排名14/100。  

## 操作流程
### 1. 環境設定 (Environment Setup)
請確保環境中安裝了必要的套件。建議使用 Python 3.8+ 與 PyTorch 2.0+。
安裝指令：
`pip install -r requirements.txt`
若需手動安裝 PyTorch 以支援 GPU，請參考 PyTorch 官網指令。

### 2. 資料準備 (Data Preparation)
確保目錄結構如下：
Sentiment Analysis/  
├── dataset/  
│   ├── train.csv  
│   ├── test.csv  
│   └── dataset.csv  
├── main.py  
├── model.py  
└── requirements.txt  

### 3. 執行訓練 (Training)
單一模型訓練，main.py 可以在 Colab 或本地端執行，執行指令如下：  

python main.py \     ## 若在colab，則開頭需!python  
    --train_csv ./dataset/train.csv \  
    --test_csv ./dataset/test.csv \  
    --out_dir ./saved_models_large \  
    --model_name microsoft/deberta-v3-large \  
    --max_length 128 \  
    --batch_size 8 \  
    --epochs 4 \  
    --dropout 0.1 \  
    --lr_encoder 1e-5 \  
    --lr_head 1e-4 \  
    --warmup_ratio 0.1 \  
    --seed 42
執行後即可重建訓練環境並開始訓練。  

### 4. 參數說明 (Arguments)  
|參數 |         預設值      |                 說明|
|--|---|--  |
|model_name | microsoft/deberta-v3-large  |使用的 Pretrained Model|  
|batch_size | 8                           |訓練批次大小|
|epochs     | 4                          |訓練總輪數  |
|lr_encoder | 1e-5                        |Transformer Encoder 的學習率  |
|lr_head    | 1e-4                        |Classification Head 的學習率  |
|dropout    | 0.1                         |Dropout 機率  |
|seed       | 42                          |隨機種子  |
|out_dir    | ./saved_models/             |模型與結果輸出路徑|  



### 5. 輸出結果
訓練完成後，指定的輸出資料夾 將包含：  
checkpoint/ : 儲存驗證集 (Val) 表現最好的模型權重 (pytorch_model.bin) 與 Tokenizer。  
summary.json: 包含訓練、驗證、測試集的準確度 與 hyperparameter 紀錄。  
val_cm.csv / test_cm.csv: 混淆矩陣原始數據。
val_report.txt / test_report.txt: 詳細的Report，含 Precision, Recall, F1-score。









