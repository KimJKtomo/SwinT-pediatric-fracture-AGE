# run_all_training.py

import os
import subprocess

# Step 1: 고정 TestSet이 없으면 생성
if not os.path.exists("age_test.csv"):
    print("📌 Step 1: Creating fixed test set (age_test.csv)...")
    subprocess.run(["python", "age_split_testset.py"], check=True)
else:
    print("✅ Step 1: age_test.csv already exists. Skipping test split.")

# Step 2: 무작위 Train/Val split 생성 (매번 실행)
print("\n📌 Step 2: Generating random train/val split...")
subprocess.run(["python", "generate_age_trainval_split.py"], check=True)

# Step 3: Age Group Classifier 학습
print("\n📌 Step 3: Training Age Group Classifier...")
subprocess.run(["python", "train_age_classifier_old.py"], check=True)

# Step 4: Age별 Fracture Classification 학습
print("\n📌 Step 4: Training Fracture Classifiers by Age Group...")
subprocess.run(["python", "train_fracture_per_agegroup.py"], check=True)

print("\n🎉 All training steps completed successfully!")
