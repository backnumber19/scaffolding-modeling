# scaffolding-modeling

스캐폴딩 작업 공수 예측을 위한 회귀 모델링 프로젝트입니다.  
원본 데이터로부터 누수 없는 전처리/분할을 수행하고, CatBoost 기반 10-fold CV 및 SHAP 중요도 분석을 제공합니다.  
Optuna 튜닝으로 하이퍼파라미터 탐색도 지원합니다.

## 설치 (git clone)

```
git clone https://github.com/backnumber19/scaffolding-modeling.git
cd scaffolding-modeling
```

## 폴더 구조

```
scaffolding-modeling/
├─ data/                   # 생성 후 원본 데이터 업로드 필요, preprocess.py로 전처리된 데이터 위치
├─ src/
│  ├─ preprocess.py        # 클렌징 및 train/test split 후 data/에 결과 저장
│  ├─ train.py             # 10-fold CV + SHAP value 산출
│  ├─ optimizer.py         # Hyperparameter optimization (by OPTUNA)
│  └─ utils.py             # 유틸성 함수(메트릭 및 logger 세팅)
├─ outputs/
│  ├─ metrics.json         # 성능 측정 결과
│  ├─ test_predictions.csv # 예측 결과
│  ├─ shap_summary.png     # SHAP value 산출 결과
│  ├─ best_params.json     # Hyperparameter optimization로 찾은 CatBoost Hyperparameter
│  └─ tuner_results.json   # Hyperparameter optimization의 best score
└─ logs/                   # log 파일
```

## Python 3.11 가상환경 생성 및 설치

`requirements.txt` 기준으로 패키지를 설치합니다.

### Windows
```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 전처리/분할 (누수 없음)

`preprocess.py`는 **클렌징 + 분할만 수행**합니다.  
스케일링/원핫 인코딩은 `train.py`의 fold 내부에서 처리됩니다.

```bash
python src/preprocess.py
  --input data/FortillsDataset_JW_cleaned.xlsx
  --target-col SumOfManhoursProrate
  --index-col TaskID
  --test-size 0.2
  --seed 42
  --out-dir data
  --log-dir logs
```

## 학습 (10-fold CV + SHAP)

```bash
python src/train.py
  --data-dir data
  --out-dir outputs
  --log-dir logs
  --index-col TaskID
  --seed 42
  --folds 10
  --iterations 2000
  --lr 0.05
  --depth 8
  --verbose 200
```

### 성능 측정 방식

- `test_mean`: 10개 fold 모델의 **test 예측 평균** 성능  
- `test_full_model`: 전체 train으로 학습한 **단일 모델** 성능  
- **full_model R²가 더 높으면** `test_predictions.csv`는 full_model 예측으로 저장됩니다.

### 모델 저장

`train.py`는 기본적으로 `outputs/full_model.cbm`에 모델을 저장합니다.

```bash
python src/train.py --model-path outputs/full_model.cbm
```

### 튜너 결과 자동 반영

`outputs/best_params.json`이 존재하면 `train.py`에서 자동으로 불러와 `cat_params`에 반영합니다.

## Optuna 튜닝

```bash
python src/optimizer.py
  --data-dir data
  --out-dir outputs
  --log-dir logs
  --index-col TaskID
  --seed 42
  --folds 10
  --trials 100
```

생성 파일:
- `outputs/best_params.json`
- `outputs/tuner_results.json`
- `logs/tuner.log`

## 출력물

- `outputs/metrics.json`  
  - `val_mean`, `test_mean`, `test_full_model` 포함
- `outputs/test_predictions.csv`  
  - `y_true`, `y_pred`
- `outputs/shap_summary.png`

## 주요 인자

- `--target-col` 기본: `SumOfManhoursProrate`
- `--index-col` 기본: `TaskID`
- `--test-size` 기본: `0.2`
- `--folds` 기본: `10`

