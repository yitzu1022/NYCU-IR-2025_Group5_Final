# Visual RAG for Geo-localization (IR Final Project)

本專案為資訊檢索與擷取 Generative Information Retrieval 課程期末Project。我們實作了一個基於 **Visual RAG (Retrieval-Augmented Generation)** 的地理定位系統。系統結合了視覺檢索模型 (CLIP / GeoCLIP) 與大型語言模型 (Google Gemini)，透過檢索相似圖片作為上下文，輔助 LLM 進行更精準的地理位置推論。

專案完整書面報告請參考：
[Project Report](docs/project_report.pdf)

## Project Structure

- **`ir_final.py`**: 主要執行檔。包含資料前處理、索引建置 (Indexing)、檢索 (Retrieval) 及 LLM 生成 (Generation) 的完整流程。
- **`geolocation-geoguessr-images-50k/`**: (需自行建立) 存放測試圖片與 Knowledge Base 圖片的資料夾。
- **`checkpoints/`**: 存放 GeoCLIP 或其他模型權重的目錄。

## Core Technologies

本系統採用 `UnifiedGeoRAG` 架構，整合以下技術：

1.  **Retrieval (檢索器)**:
    * **Embeddings**: 支援 **OpenAI CLIP (ViT-B/32)** 與 **GeoCLIP** (Location-Aware Image Encoding)。
    * **Vector DB**: 使用 **FAISS** 進行高效率的向量相似度搜尋 (IndexFlatL2)。
    * **Geography**: 利用 `geopy` 計算地理座標距離 (Geodesic distance)。

2.  **Generation (生成器)**:
    * **Models**: 
      - **Gemini 2.0 Flash Exp** 
      - **Gemini 2.5 Flash Image** 
    * **Strategy**: 將檢索到的 Top-K 相似圖片及其 metadata (經緯度、地點描述) 作為 Prompt Context，引導 LLM 進行多模態推理，輸出預測地點。

## Dataset

- **來源**: GeoGuessr 街景截圖資料集 (~50,000 張圖片，涵蓋 150+ 國家)
- **資料分割**: 
  - Knowledge Base (訓練集): 90%
  - Test Set: 10% (實驗中使用 50 個樣本)
- **標註**: 每張圖片標註所屬國家及該國中心點經緯度

## Environment Setup

### 1. Prerequisites
* Python 3.8+
* CUDA 支援 (建議使用 NVIDIA GPU 以加速 Embedding 計算與 FAISS 檢索)
* Google Gemini API Key

### 2. Install Packages
請執行以下指令安裝所需套件：

```bash
pip install -r requirements.txt
```

**注意**：`faiss-gpu` 僅適用於 Linux/Windows 的 NVIDIA GPU 環境。若使用 MacOS 或無 GPU，請將 requirements.txt 中的 `faiss-gpu` 改為 `faiss-cpu`。

### 3. Main Packages
- `torch`, `clip`, `faiss-gpu`
- `google-generativeai` (Gemini API)
- `geopy`, `Pillow`, `pandas`, `numpy`
- `geoclip` (選用，用於 GeoCLIP 編碼器)

## Usage Instructions

### 1. Set API Key
請在環境變數中設定您的 Google API Key，或直接於程式碼中填入：

```python
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
```

### 2. Prepare Dataset
確保 GeoGuessr 資料集已下載並放置於指定路徑（預設為 `./geolocation-geoguessr-images-50k`）。

Structure：
```
geolocation-geoguessr-images-50k/
├── United States/
│   ├── image1.jpg
│   └── image2.jpg
├── Japan/
│   ├── image1.jpg
│   └── image2.jpg
└── ...
```

### 3. Run Experiments

```python
# 初始化系統 (選擇 CLIP 或 GeoCLIP)
system = UnifiedGeoRAG(encoder_type="GeoCLIP")

# 建立 FAISS 索引
system.build_index(kb_data)

# 執行完整實驗套件
df_results = run_experiments()

# 生成分析報告
print_summary_tables(df_results)
```

### 4. Experiment Modes
系統支援三種實驗模式（可在 CONFIG 中切換）：

- **Gemini Only** (`RUN_GEMINI_ONLY`): Zero-shot 地理定位基準測試
- **KNN** (`RUN_KNN`): 基於向量檢索的 k-NN 分類（支援 Voting / Weighted Average）
- **RAG** (`RUN_RAG`): 完整 Visual RAG Pipeline（檢索 + LLM 推理）

### 5. Qualitative Analysis Mode
除了量化評估，系統還提供旅遊導覽生成功能：

```python
run_qualitative_showcase(system, test_data, num_samples=3, k=3)
```

此模式會生成：
- 地點預測與推理過程
- 視覺線索分析（建築、植被、路標等）
- 旅遊導覽文字（附近景點推薦）

## System Configuration Parameters

### Main CONFIG Settings

```python
CONFIG = {
    # API Configuration
    "GEMINI_API_KEY": "YOUR_API_KEY",
    "MODEL_NAME": "gemini-2.0-flash-exp",  # 或 "gemini-2.5-flash-image"
    
    # Path Configuration
    "DATASET_PATH": "./geolocation-geoguessr-images-50k",
    "INDEX_DIR": "./geo_indexes",
    
    # Model Configuration
    "CLIP_MODEL": "ViT-B/32",
    "DEVICE": "cuda",  # 或 "cpu"
    
    # Experiment Configuration
    "SEED": 23,
    "TEST_SAMPLE_SIZE": 50,
    "RETRY_DELAY": 20,
    "MAX_RETRIES": 3,
    
    # Experiment Modes
    "RUN_GEMINI_ONLY": True,  # Zero-shot baseline
    "RUN_KNN": True,          # k-NN retrieval
    "RUN_RAG": True,          # Full RAG pipeline
    
    # RAG Configuration
    "RAG_SAMPLES_PER_K": {
        1: 50,   # K=1 使用 50 個測試樣本
        3: 50,   # K=3 使用 50 個測試樣本
        5: 50,   # K=5 使用 50 個測試樣本
        10: 50   # K=10 使用 50 個測試樣本
    }
}
```

### Encoder Selection

系統支援兩種視覺編碼器：

1. **Standard CLIP (ViT-B/32)**
   ```python
   system = UnifiedGeoRAG(encoder_type="CLIP")
   ```
   - 通用視覺特徵編碼器
   - 預訓練於大規模圖文配對資料
   - 無需額外安裝

2. **GeoCLIP**
   ```python
   system = UnifiedGeoRAG(encoder_type="GeoCLIP")
   ```
   - 針對地理定位任務微調
   - 捕捉地理特定視覺特徵（植被、建築風格、道路標示等）
   - 需額外安裝: `pip install geoclip`

### FAISS Index Configuration

- **索引類型**: `IndexFlatL2` (精確 L2 距離搜尋)
- **向量維度**: 
  - CLIP: 512 維
  - GeoCLIP: 模型預設維度
- **檢索策略**: Top-K 最近鄰搜尋

### Evaluation Metrics

系統使用以下指標評估效能：

1. **Country Prediction Accuracy**: 國家預測準確率
   ```
   Accuracy = (正確預測數) / (總測試樣本數)
   ```

2. **Mean Distance Error (MDE)**: 平均距離誤差
   ```
   MDE = Σ distance(pred_center, truth_center) / N
   ```
   - 計算預測國家中心點與真實國家中心點的 Haversine 距離

3. **Rank@K**: 檢索品質評估
   ```
   Rank@K = (正確國家出現在 Top-K 檢索結果中的比例)
   ```

4. **Accuracy @ Distance Thresholds**: 不同距離閾值下的準確率
   - City Level: 25 km
   - Region Level: 200 km
   - Country Level: 750 km
   - Continent Level: 2500 km

## Data Preprocessing

### Image Loading and Coordinate Annotation

```python
def load_dataset_with_coords(root_path: str) -> List[Dict]:
    """
    載入資料集並標註國家中心點座標
    
    Returns:
        List[Dict]: 包含以下欄位
            - image_path: 圖片路徑
            - country: 國家名稱
            - latitude: 國家中心點緯度
            - longitude: 國家中心點經度
    """
```

處理流程：
1. 掃描資料夾結構，提取國家名稱
2. 使用 `geopy.Nominatim` 查詢國家中心點座標
3. 快取座標結果以加速載入
4. 每個國家限制載入前 100 張圖片（可調整）

### FAISS Index Building

```python
def build_index(self, data_list, batch_size=32):
    """
    建立 FAISS 向量索引
    
    Args:
        data_list: 資料清單
        batch_size: 批次處理大小（建議 32-64）
    
    Process:
        1. 批次編碼所有圖片
        2. 堆疊特徵向量為矩陣
        3. 建立 IndexFlatL2 索引
        4. 儲存對應的 metadata
    """
```

## Experiment Mode Descriptions

### 1. Gemini Only (Zero-shot Baseline)

直接使用 Gemini 視覺模型進行地理定位，無檢索輔助。

```python
CONFIG["RUN_GEMINI_ONLY"] = True
```

**Prompt 結構**:
```
Predict the geographic location of this image.

Output format (strictly follow):
Latitude: [your prediction]
Longitude: [your prediction]
Country: [your prediction]
```

### 2. k-NN Retrieval (Baseline)

基於向量相似度的 k-近鄰分類。

```python
CONFIG["RUN_KNN"] = True
```

**支援的聚合策略**:
- `nearest` (K=1): 使用最近鄰的標籤
- `vote`: 多數投票 + 中位數座標
- `weighted`: 距離加權平均

```python
def run_knn_logic(retrieved_items, k, mode='weighted'):
    """
    Args:
        retrieved_items: 檢索結果
        k: 近鄰數量
        mode: 'nearest', 'vote', 'weighted'
    """
```

### 3. RAG (Retrieval-Augmented Generation)

完整的檢索增強生成流程。

```python
CONFIG["RUN_RAG"] = True
```

**Pipeline**:
1. **Retrieval**: 檢索 Top-K 相似圖片
2. **Context Construction**: 建構包含範例圖片的 Prompt
3. **Generation**: Gemini 多模態推理

**Prompt 結構**:
```python
def build_rag_prompt(retrieved_items):
    """
    建構 RAG Prompt，包含：
    - 任務指示
    - K 個檢索範例（圖片 + 座標 + 國家）
    - 查詢圖片
    - 輸出格式要求
    """
```

### 4. Qualitative Showcase

生成描述性分析與旅遊導覽。

```python
run_qualitative_showcase(system, test_data, num_samples=3, k=3)
```

**輸出內容**:
- 地點預測（經緯度、國家）
- 視覺推理解釋（建築、植被、路標等線索）
- 旅遊導覽段落（景點介紹、推薦）

## API Calls and Error Handling

### Gemini API Retry Mechanism

```python
def generate_gemini_response(self, prompt, images):
    """
    呼叫 Gemini API（包含重試機制）
    
    Features:
        - 自動重試（最多 3 次）
        - 指數退避策略（20s, 40s, 80s）
        - Rate limit 檢測與等待
        - 錯誤日誌記錄
    """
```

### Rate Limit Management

- 每次 RAG 呼叫後等待 3 秒
- Gemini Only 模式每次呼叫後等待 2 秒
- 429 錯誤觸發指數退避

## Output and Result Storage

### CSV Result Files

系統自動儲存實驗結果：

```
full_experiment_results_YYYYMMDD_HHMMSS.csv
```

**欄位說明**:
- `Sample_ID`: 測試樣本編號
- `Method`: 方法名稱（Gemini Only / KNN / RAG）
- `Encoder`: 編碼器類型（CLIP / GeoCLIP）
- `K`: 檢索數量
- `Aggregation`: 聚合策略
- `Distance_KM`: 距離誤差（公里）
- `Country_Match`: 國家匹配（True/False）
- `Model_Version`: Gemini 模型版本

### Summary Statistics

```python
def print_summary_tables(df):
    """
    生成摘要報表：
    1. 平均距離誤差（按方法分組）
    2. 不同距離閾值下的準確率
    3. Top 5 最佳方法排名
    """
```

## Code Structure

```
ir_final.py
├── Configuration (CONFIG)
├── Data Loading
│   └── load_dataset_with_coords()
├── UnifiedGeoRAG Class
│   ├── __init__()
│   ├── _load_encoder()
│   ├── build_index()
│   ├── retrieve()
│   └── generate_gemini_response()
├── Evaluation Functions
│   ├── parse_coords()
│   ├── calculate_error()
│   ├── run_knn_logic()
│   └── build_rag_prompt()
├── Experiment Execution
│   ├── log_result()
│   ├── run_experiments()
│   └── run_experiments_v2()
├── Analysis & Visualization
│   ├── print_summary_tables()
│   └── run_qualitative_showcase()
└── Main Execution
```

## Experimental Results

### Main Method Comparison

| Method | Country Accuracy (%) ↑ | Mean Distance Error (km) ↓ |
|--------|------------------------|------------------------------|
| DenseNet121 (Baseline) | 34.23 | - |
| KNN w/ CLIP | 34 | 2979.32 |
| KNN w/ GeoCLIP | 56 | 1311.11 |
| Gemini 2.5 (Zero-shot) | 60 | 841.67 |
| **UnifiedGeoRAG (Ours)** | **64** | **1455.52** |

Note: UnifiedGeoRAG uses GeoCLIP encoder with K=3 retrieval configuration. For complete experimental results and analysis, please refer to `Project_Team5_report.pdf`.

## System Output Example

```
Query Image: [圖片]

Retrieved Context:
  1. Taiwan, Hsinchu (Dist: 5.2 km) - [參考圖片]
  2. Taiwan, Taichung (Dist: 12.8 km) - [參考圖片]
  3. Taiwan, Taipei (Dist: 18.3 km) - [參考圖片]

Gemini Reasoning:
"根據檢索到的圖片，建築風格呈現現代化設計，路標使用繁體中文，
道路兩旁種植亞熱帶植物。參考 Example 1 的地理座標，推測此地點
位於台灣新竹市附近的科技園區..."

Final Prediction:
  Latitude: 24.7936
  Longitude: 120.9962
  Country: Taiwan
```

## Future Work

1. **多模態嵌入融合**: 整合經緯度座標、國家標籤、氣候描述等 metadata 到 embedding 空間
2. **階層式檢索**: 先預測洲/國，再細化到城市/地標
3. **增量學習**: 支援新國家/地區的動態加入，無需重新訓練
4. **使用者回饋機制**: 整合人類回饋以優化檢索排序

## Team Members

- 413551036 翁祥恩
- 313551073 顏琦恩
- 314551058 唐文蔚
- 314554006 陳乙慈

## References

1. [Img2Loc (SIGIR 2024)](https://arxiv.org/abs/2405.04793) - Geolocation via Retrieval-Augmented Generation
2. [DQU-CIR (SIGIR 2024)](https://arxiv.org/abs/2405.08706) - CLIP-based Visual Retrieval
3. [IM-RAG (SIGIR 2024)](https://arxiv.org/abs/2405.12021) - Multi-round Retrieval-Augmented Generation
4. [CLIP (OpenAI)](https://github.com/openai/CLIP) - Contrastive Language-Image Pre-training
5. [GeoCLIP](https://github.com/VicenteVivan/geo-clip) - Location-Aware Image Encoder

---

**Last Updated**: December 2024  
**Course**: Information Retrieval Final Project  
**Institution**: National Yang Ming Chiao Tung University
