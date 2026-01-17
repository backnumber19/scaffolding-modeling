# scaffolding-modeling

스캐폴딩 작업 공수 예측을 위한 CatBoost 회귀 모델링 프로젝트입니다.

## 프로젝트 개요

- **모델**: CatBoost Regressor
- **검증**: K-fold Stratified Cross-Validation
- **전처리**: 데이터 누수 방지를 위해 fold 내부에서 스케일링/인코딩 수행
- **튜닝**: Optuna 기반 하이퍼파라미터 최적화
- **해석**: SHAP 기반 변수 중요도 분석

---

## 폴더 구조

```
scaffolding-modeling/
├─ data/                       # 원본 데이터(직접 업로드 필요) 및 preprocess.py의 결과물
├─ src/
│  ├─ preprocess.py            # 데이터 클렌징 + train/test 분할
│  ├─ train.py                 # 10-fold CV 학습 + SHAP 분석
│  ├─ optimizer.py             # Optuna 하이퍼파라미터 튜닝
│  └─ utils.py                 # 유틸리티 함수 (Metric, Logger 구현현)
├─ outputs/
│  ├─ metrics.json             # 성능 측정 결과 (train.py의 결과물)
│  ├─ test_predictions.csv     # 테스트 예측 결과 (train.py의 결과물)
│  ├─ shap_summary.png         # SHAP 변수 중요도 플롯 (train.py의 결과물)
│  ├─ best_params.json         # 튜닝된 하이퍼파라미터 (optimizer.py의 결과물)
│  ├─ tuner_results.json       # 튜닝 결과 요약 (optimizer.py의 결과물)
│  └─ full_model.cbm           # 전체 학습 데이터 기반 최종 모델 (train.py의 결과물)
├─ logs/                       # 로그 파일
├─ requirements.txt            # 의존성 패키지 목록
└─ README.md
```

---

## 🚀 Quick Start (전체 워크플로우)

### Step 0. 프로젝트 클론

```bash
git clone https://github.com/backnumber19/scaffolding-modeling.git
cd scaffolding-modeling
```

---

### Step 1. 가상환경 생성 및 패키지 설치

#### Windows (PowerShell)

```powershell
python -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2. 원본 데이터 준비

`data/` 폴더에 원본 엑셀 파일을 넣습니다:

---

### Step 3. 전처리 (데이터 클렌징 + Train/Test 분할)

```bash
python src/preprocess.py \
  --input {$YOUR_RAW_DATA_PATH} \
  --target-col SumOfManhoursProrate \
  --index-col TaskID \
  --test-size 0.2 \
  --seed 42 \
  --out-dir data \
  --log-dir logs
```

**출력 파일:**
- `data/train.xlsx`, `data/test.xlsx`
- `data/train_target.xlsx`, `data/test_target.xlsx`
- `logs/preprocess.log`

> ⚠️ 스케일링/원핫 인코딩은 여기서 하지 않습니다 (데이터 누수 방지)

---

### Step 4. 초기 학습 (Baseline)

튜닝 없이 기본 파라미터로 먼저 학습합니다:

```bash
python src/train.py \
  --data-dir data \
  --out-dir outputs \
  --log-dir logs \
  --index-col TaskID \
  --seed 42 \
  --folds 10 \
  --iterations 2000 \
  --lr 0.05 \
  --depth 8 \
  --verbose 200
```

**출력 파일:**
- `outputs/metrics.json` — 성능 지표 (R², Adj.R², MAE, RMSE, RAE)
- `outputs/test_predictions.csv` — 테스트 예측 결과
- `outputs/shap_summary.png` — SHAP 변수 중요도
- `outputs/full_model.cbm` — 저장된 모델
- `logs/train.log`

---

### Step 5. 하이퍼파라미터 튜닝 (Optuna)

Optuna로 최적의 하이퍼파라미터를 탐색합니다:

```bash
python src/optimizer.py \
  --data-dir data \
  --out-dir outputs \
  --log-dir logs \
  --index-col TaskID \
  --seed 42 \
  --folds 10 \
  --trials 100
```

**출력 파일:**
- `outputs/best_params.json` — 최적 하이퍼파라미터
- `outputs/tuner_results.json` — 튜닝 결과 요약
- `logs/tuner.log`

> 💡 `--trials` 값을 늘리면 더 많은 조합을 탐색합니다 (시간 증가)

---

### Step 6. 최종 학습 (튜닝된 파라미터 적용)

`best_params.json`이 존재하면 자동으로 로드됩니다(없으면 기본 하이퍼파라미터 적용):

```bash
python src/train.py \
  --data-dir data \
  --out-dir outputs \
  --log-dir logs \
  --index-col TaskID \
  --seed 42 \
  --folds 10 \
  --verbose 200
```

> `train.py`는 `outputs/best_params.json`을 자동 감지하여 적용합니다.

---

## 📊 성능 측정 방식

`metrics.json`에는 세 가지 성능이 기록됩니다:

| 항목 | 설명 |
|------|------|
| `val_mean` | 10-fold validation 평균 성능 |
| `test_mean` | 10개 fold 모델의 test 예측 앙상블 성능 |
| `test_full_model` | 전체 train 데이터로 학습한 단일 모델 성능 |

- **full_model R²가 더 높으면** → `test_predictions.csv`에 full_model 예측 저장
- **그렇지 않으면** → 앙상블 평균 예측 저장

---

## 주요 인자 정리

### preprocess.py

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input` | `data/FortillsDataset_JW_cleaned.xlsx` | 원본 데이터 경로 |
| `--target-col` | `SumOfManhoursProrate` | 타겟 컬럼명 |
| `--index-col` | `TaskID` | 인덱스 컬럼명 |
| `--test-size` | `0.2` | 테스트 비율 |
| `--seed` | `42` | 랜덤 시드 |

### train.py

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data-dir` | `data` | 전처리된 데이터 경로 |
| `--out-dir` | `outputs` | 출력 경로 |
| `--folds` | `10` | CV fold 수 |
| `--iterations` | `2000` | CatBoost iterations |
| `--lr` | `0.05` | learning rate |
| `--depth` | `8` | tree depth |
| `--model-path` | `outputs/full_model.cbm` | 모델 저장 경로 |

### optimizer.py

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--trials` | `100` | Optuna 탐색 횟수 |
| `--folds` | `10` | CV fold 수 |

---

## 기술적 특징

1. **데이터 누수 방지**: 스케일링/원핫 인코딩은 각 fold 내부에서 fit → transform
2. **타겟 변환**: `log1p(y)` 변환 후 학습, 예측 시 `expm1(pred)` 역변환
3. **손실 함수**: MAE (이상치에 강건)
4. **CV 전략**: StratifiedKFold (타겟 분포 균등화)
5. **병렬 튜닝**: Optuna `n_jobs=-1`로 멀티코어 활용
