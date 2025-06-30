# 🦴 Age Group-wise Fracture Classification using Swin Transformer

이 프로젝트는 소아 및 청소년 X-ray 데이터를 나이 그룹별로 분류하여, 각 그룹에 최적화된 골절 분류 모델을 학습하는 파이프라인입니다. Swin Transformer (`timm`) 백본을 사용하여 정밀한 분류 성능을 도모합니다.

---

## 📁 파이프라인 구조

```
.
├── train_fracture_per_agegroup.py     # 나이 그룹별 골절 분류기 학습 스크립트
├── age_train_tmp.csv                  # 나이 분류기 예측 기반의 학습용 데이터 (age_group_label 포함)
├── age_val_tmp.csv                    # 나이 분류기 예측 기반의 검증용 데이터
├── saved_agegroup_models/             # 그룹별로 학습된 모델 저장 경로
├── unified_dataset_0605.py            # 공통 Dataset 정의
└── ...
```

---

## 🔁 파이프라인 개요

1. **age\_train\_tmp.csv / age\_val\_tmp.csv 생성**

   * 나이 분류기 (`train_age_classifier.py`)를 학습 및 추론하여 생성
   * `age_group_label` 열을 기준으로 3개 그룹으로 나눔 (예: 0: 0-7세, 1: 8-14세, 2: 15-19세)

2. **train\_fracture\_per\_agegroup.py 실행**

   * 각 나이 그룹별로 데이터를 분리하여 골절 여부(0/1)를 분류하는 Swin Transformer 모델을 학습
   * 학습에는 `timm` 라이브러리 기반 Swin-L 사용
   * `FocalLoss` 및 `WeightedSampler`로 클래스 불균형 보정
   * EarlyStopping과 Confusion Matrix / F1-score 모니터링 포함

---

## 🧠 모델 구조

* 백본: `swin_large_patch4_window12_384_in22k` (`timm`)
* 출력: Binary classification (Fracture / Normal)
* 손실 함수: FocalLoss (α=0.25, γ=2.0)
* Optimizer: AdamW (lr=1e-5)

---

## 📊 출력 예시

```
🔹 Training for Age Group 1...
🖼️ Total images - Train: 5030 | Val: 1276

[Train] Loss: 64.1092 | Acc: 0.7921 | F1: 0.8055
[[1405  604]
 [ 434 2587]]

[Val] Acc: 0.7614 | F1: 0.7782
[[346 232]
 [ 70 628]]
```

---

## 💾 결과 저장

* 학습된 모델은 다음 경로에 저장됩니다:

  * `saved_agegroup_models/fracture_classifier_agegroup0.pt`
  * `saved_agegroup_models/fracture_classifier_agegroup1.pt`
  * `saved_agegroup_models/fracture_classifier_agegroup2.pt`

---

## ⚙️ 실행 방법

```bash
python train_fracture_per_agegroup.py
```

사전에 `age_train_tmp.csv`, `age_val_tmp.csv`가 생성되어 있어야 합니다.

---

## 📦 Dependencies

* Python 3.8+
* PyTorch
* `timm`
* `scikit-learn`
* `pandas`, `numpy`, `tqdm`

---

## 📬 문의

의료영상 기반 AI 모델 개발 관련 문의는 개발자에게 연락 주세요.

