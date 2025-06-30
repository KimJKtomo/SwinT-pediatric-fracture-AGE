# 🦴 Age Group-wise Fracture Classification using Swin Transformer

이 프로젝트는 소아 및 청소년 X-ray 데이터를 대상으로, 나이 그룹을 분류한 후 각 그룹별로 최적화된 골절 분류 모델을 학습하는 전체 자동화 파이프라인입니다. Swin Transformer (`timm`) 백본을 활용하여 정밀한 골절 분류 성능을 구현합니다.

---

## 📁 전체 프로젝트 구조

```
.
├── run_all_training.py                # 전체 파이프라인 자동 실행 메인 스크립트
├── age_split_testset.py              # 고정 Test 세트 분리 스크립트
├── generate_age_trainval_split.py    # Train/Val 랜덤 분리 스크립트
├── train_age_classifier_old.py       # 나이 분류기 학습 스크립트
├── train_fracture_per_agegroup.py    # 그룹별 골절 분류기 학습 스크립트
├── age_test.csv                      # 고정 Test 세트 결과
├── age_train_tmp.csv                 # 예측된 나이 그룹이 포함된 Train 세트
├── age_val_tmp.csv                   # 예측된 나이 그룹이 포함된 Val 세트
├── saved_agegroup_models/            # 학습된 모델 저장 폴더
├── unified_dataset_0605.py           # 공통 Dataset 정의 스크립트
└── ...
```

---

## 🔁 전체 파이프라인 개요

1. **Test Set 분리 (**\`\`**)**

   * 전체 원본 데이터에서 고정된 test set (`age_test.csv`)를 생성

2. **Train/Val Split 생성 (**\`\`**)**

   * test를 제외한 나머지 샘플을 기준으로 `age_train_tmp.csv`, `age_val_tmp.csv` 분리 생성

3. **나이 그룹 분류기 학습 (**\`\`**)**

   * Age(연령)를 3개 그룹으로 분류 (0: 0–7세, 1: 8–14세, 2: 15–19세)
   * 예측 결과를 `age_group_label` 열로 저장

4. **나이 그룹별 골절 분류기 학습 (**\`\`**)**

   * 각 `age_group_label` 그룹을 기준으로 데이터를 분할
   * Swin Transformer (`timm`) 기반 모델을 학습
   * `FocalLoss`, `WeightedSampler`, EarlyStopping 전략 적용

---

## 🧠 모델 구조 및 학습 설정

* 백본: `swin_large_patch4_window12_384_in22k` (`timm`)
* 출력: Binary Classification (Fracture / Normal)
* 손실 함수: FocalLoss (α=0.25, γ=2.0)
* Optimizer: AdamW (lr=1e-5)
* 평가 지표: Accuracy, F1-score, Confusion Matrix

---

## 📊 학습 출력 예시

### 🔹 Age Classification

```
[Epoch 13] 📊 Val Confusion Matrix:
[[ 668  131    0]
 [  80 2067   82]
 [   0  141  641]]

              precision    recall  f1-score   support

           0     0.8930    0.8360    0.8636       799
           1     0.8837    0.9273    0.9050      2229
           2     0.8866    0.8197    0.8518       782

    accuracy                         0.8861      3810
   macro avg     0.8878    0.8610    0.8735      3810
weighted avg     0.8863    0.8861    0.8854      3810
```

### 🔹 Age Group 1 (8–14세)

```
🖼️ Total images - Train: 2600 | Val: 657

[Epoch 0] Train Loss: 248.78 | Acc: 0.7262 | F1: 0.7807
Val Acc: 0.7282 | F1: 0.7740
[[ 160  143]
 [  35  319]]

[Epoch 1] Train Loss: 231.21 | Acc: 0.7447 | F1: 0.7923
Val Acc: 0.7527 | F1: 0.7857
[[ 163  140]
 [  23  331]]
```

---

## 💾 모델 저장 위치

각 그룹별 학습이 완료되면 다음 경로에 모델이 저장됩니다:

```
saved_agegroup_models/
├── fracture_classifier_agegroup0.pt
├── fracture_classifier_agegroup1.pt
└── fracture_classifier_agegroup2.pt
```

---

## ⚙️ 실행 방법

전체 파이프라인은 아래 명령으로 한 번에 실행됩니다:

```bash
python run_all_training.py
```

### 내부 실행 순서:

1. `age_split_testset.py` – Test set 분리
2. `generate_age_trainval_split.py` – Train/Val 분리
3. `train_age_classifier_old.py` – 나이 그룹 분류기 학습
4. `train_fracture_per_agegroup.py` – 그룹별 골절 분류기 학습

---

## 📦 Dependencies

환경은 `swin_env.yaml` 기준이며 주요 패키지는 다음과 같습니다:

* Python 3.7
* PyTorch 1.13.1
* torchvision 0.14.1
* timm 0.4.12
* scikit-learn 1.0.2
* albumentations 1.3.1
* pandas, numpy, tqdm
* grad-cam, matplotlib, seaborn
* torchsampler, openpyxl, yacs, opencv-python

자세한 패키지는 `swin_env.yaml` 참고.

---

## 📬 문의

의료 영상 기반 AI 시스템 개발 관련 문의는 프로젝트 관리자에게 연락 주세요.

