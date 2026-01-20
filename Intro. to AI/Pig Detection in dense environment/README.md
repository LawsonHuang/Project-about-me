# Pig Detection in Dense Environments (HW4_111511167)
本專案實作基於 YOLO11x 的豬隻偵測模型，針對密集遮擋與多視角環境進行優化。以下說明包含環境建置、資料前處理、模型訓練與評估流程。  

## Project Structure 

請確保您的目錄結構如下所示，以利程式順利執行：

HW4_111511167/
├── Dataset/
│   ├── merge_sort.py           # 資料集重組與清洗腳本
│   ├── train/                  # 原始訓練集資料夾
│   │   ├── images/
│   │   └── labels/
│   └── test/                   # 原始測試集資料夾 (或是隱藏測試集)
│       ├── images/
│       └── labels/
├── 111511167.pt                # 預訓練好的最佳權重 (For Evaluation)
├── train.py                    # 模型訓練主程式
├── test.py                     # 模型推論與評估程式
├── requirements.txt            # 環境相依套件清單
└── README.md                   # 使用說明文件





Environment Settings : 

本專案建議於 Google Colab 環境下執行，使用 GPU 加速。

Hardware: NVIDIA L4 GPU (Training) / T4 GPU (Inference)

Memory: 25GB RAM (High-RAM mode recommended)

安裝相依套件
在開始之前，請先安裝必要的 Python 套件：
pip install -r requirements.txt
(註：若在 Colab Notebook 中執行，請在指令前加上 !，例如 !pip install ...)





How to Run Training (此為最終繳交的權重訓練方式，若需要文章內前半段的訓練方式，請關注文章末端說明): 

為了重現最佳效能，訓練前必須執行資料集重分配。請依序執行以下步驟：

Step 1: 資料前處理 (目前Dataset沒有image與labels，需要自行添加)
進入 Dataset 資料夾並執行重組腳本，此步驟將合併原始資料並重新隨機分配以解決分佈不均問題。
1. cd Dataset
2. python merge_sort.py

Step 2: 資料夾更名
腳本執行完畢後，會生成 train_new 與 test_new 資料夾。請手動執行以下操作：
1. 刪除 (或備份) 原有的 train 與 test 資料夾。
2.將 train_new 重新命名為 train。
3.將 test_new 重新命名為 test。

Step 3: 執行訓練 (Start Training)
回到專案根目錄，執行訓練腳本：
1. cd ..
2. python train.py

Step 4: 取得權重 (Model Weights)
訓練完成後，最佳權重將儲存於以下路徑：
Path: runs/detect/train_pig_yolo11_x_1280_MAP50/weights/best.pt
請將此檔案複製回根目錄並重新命名為 111511167.pt 以供使用。





How to Run Evaluation : 
此步驟將載入訓練好的權重，並針對 Dataset/test 資料夾內的圖片進行推論與指標計算。

Step 1: 準備權重與資料
確保根目錄下已有訓練好的權重檔 111511167.pt。

若要測試 隱藏測試集 (Hidden Dataset)，請刪除原有Dataset裡面的test，並新增您的隱藏test資料夾

Step 2: 執行評估腳本
執行以下指令即可獲得 mAP、Precision、Recall 等統計數據及視覺化結果：
python test.py
輸出結果 (Outputs)
終端機輸出: 將顯示 Precision, Recall, F1-Score, mAP50, mAP50-95 等數值。
視覺化圖片: 推論後的標註圖片將儲存於 runs/detect/visualizations/ 資料夾中。





Special Notes : 
Colab 路徑: 若使用 Google Drive 掛載，請確保 cd 到正確的專案路徑後再執行指令。
OOM 問題: train.py 預設使用 Batch Size = 4 
絕對/相對路徑: 所有腳本皆使用相對路徑設計，請勿隨意移動 train.py 或 test.py 的位置，以免找不到 Dataset。












Ps. 接下來為 最初訓練方式(雖然報告內容主要提到它，但它不是最終訓練方法)

請依序執行以下步驟：

Step 1: 修改train.py參數
進入train.py，修改註解的參數(數值已顯示於註解)

Step 2: 安裝相依套件
安裝必要的 Python 套件：
pip install -r requirements.txt

Step 3: 執行訓練 (Start Training)
執行訓練腳本：
2. python train.py

Step 4: 取得權重 (Model Weights)
訓練完成後，最佳權重將儲存於以下路徑：
Path: runs/detect/train_pig_yolo11_x_1280_MAP50/weights/best.pt

請將此檔案複製回根目錄並重新命名為 111511167.pt 以供使用。
