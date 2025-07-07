import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from timm import create_model
import torchvision.transforms as transforms
from unified_dataset_0704 import UnifiedDataset
import mlflow
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 하이퍼파라미터
BATCH_SIZE = 16
EPOCHS = 40
MODEL_NAME = "swin_large_patch4_window12_384_in22k"

# ✅ Age Group 라벨링 함수 (태아기~18개월 / 18개월~4세 / 5세~9세 / 10세~14세 /15세~19세)
def AGE_GROUP_FN(age):
    age = float(age)
    if age < 1.5:
        return 0  # 태아기~18개월
    elif age < 5:
        return 1  # 18개월~4세
    elif age < 10:
        return 2  # 5세~9세
    elif age < 15:
        return 3  # 10세~14세
    else:
        return 4  # 15세~19세

# ✅ 데이터 로딩
df_train = pd.read_csv("age_train_tmp.csv")
df_val = pd.read_csv("age_val_tmp.csv")

df_train["label"] = df_train["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)
df_val["label"] = df_val["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)
df_train["age_group_label"] = df_train["age"].astype(float).apply(AGE_GROUP_FN)
df_val["age_group_label"] = df_val["age"].astype(float).apply(AGE_GROUP_FN)

# ✅ Transform 정의
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

for age_group in [0, 1, 2, 3, 4]:
    print(f"\n🔹 Training for Age Group {age_group}...")

    df_train_g = df_train[df_train["age_group_label"] == age_group]
    df_val_g = df_val[df_val["age_group_label"] == age_group]

    train_dataset = UnifiedDataset(df_train_g, transform=transform, task="fracture_only")
    val_dataset = UnifiedDataset(df_val_g, transform=transform, task="fracture_only")

    train_labels = df_train_g["label"].tolist()
    count_0, count_1 = train_labels.count(0.0), train_labels.count(1.0)
    weight_0, weight_1 = 1.0 / count_0, 1.0 / count_1
    sample_weights = [weight_0 if l == 0.0 else weight_1 for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(MODEL_NAME, pretrained=True, num_classes=1).to(DEVICE)
    pos_weight = torch.tensor([count_0 / count_1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    with mlflow.start_run(run_name=f"AgeGroup{age_group}_Weighted"):
        for epoch in range(EPOCHS):
            print(f"\n🔁 Epoch {epoch}")
            model.train()
            total_loss, preds, labels = 0, [], []

            for images, targets in train_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE).float().view(-1)

                optimizer.zero_grad()
                outputs = model(images).squeeze(dim=-1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                preds += (probs > 0.5).astype(int).tolist()
                labels += targets.cpu().numpy().astype(int).tolist()

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            print(f"[Train] Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
            print(confusion_matrix(labels, preds))
            print(f"[Train] Sigmoid Range: min={min(probs):.4f}, max={max(probs):.4f}, mean={probs.mean():.4f}")

            # ✅ Validation
            model.eval()
            val_preds, val_labels, val_probs = [], [], []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(DEVICE)
                    targets = targets.to(DEVICE).float().view(-1)
                    outputs = model(images).squeeze(dim=-1)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    val_probs += probs.tolist()
                    val_preds += (probs > 0.5).astype(int).tolist()
                    val_labels += targets.cpu().numpy().astype(int).tolist()

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            print(f"[Val] Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            print(confusion_matrix(val_labels, val_preds))
            print(f"[Val] Sigmoid Range: min={min(val_probs):.4f}, max={max(val_probs):.4f}, mean={np.mean(val_probs):.4f}")

            mlflow.log_metric("train_acc", acc, step=epoch)
            mlflow.log_metric("train_f1", f1, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

        model_path = f"0704_fracture_classifier_agegroup{age_group}.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"✅ Saved model: {model_path}")
