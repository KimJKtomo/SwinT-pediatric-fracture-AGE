import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from timm import create_model
import torchvision.transforms as transforms
from unified_dataset_0704 import UnifiedDataset
import mlflow

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 하이퍼파라미터
BATCH_SIZE = 8
EPOCHS = 40
MODEL_NAME = "swin_large_patch4_window12_384_in22k"

# ✅ 라벨 파생 함수
AGE_GROUP_FN = lambda x: 0 if x < 8 else 1 if x < 14 else 2

# ✅ Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# ✅ 데이터 로딩
df_train = pd.read_csv("age_train_tmp.csv")
df_val = pd.read_csv("age_val_tmp.csv")

df_train["label"] = df_train["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)
df_val["label"] = df_val["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)
df_train["age_group_label"] = df_train["age"].astype(int).apply(AGE_GROUP_FN)
df_val["age_group_label"] = df_val["age"].astype(int).apply(AGE_GROUP_FN)

# ✅ Transform 정의
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for age_group in [0, 1, 2]:
    print(f"\n🔹 Training for Age Group {age_group}...")

    df_train_g = df_train[df_train["age_group_label"] == age_group]
    df_val_g = df_val[df_val["age_group_label"] == age_group]

    train_dataset = UnifiedDataset(df_train_g, transform=transform, task="fracture_only")
    val_dataset = UnifiedDataset(df_val_g, transform=transform, task="fracture_only")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(MODEL_NAME, pretrained=True, num_classes=1).to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with mlflow.start_run(run_name=f"AgeGroup{age_group}_Focal"):
        for epoch in range(EPOCHS):
            print(f"\n🔁 Epoch {epoch}")
            model.train()
            train_loss, train_preds, train_labels = 0, [], []

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).float().view(-1)

                optimizer.zero_grad()
                outputs = model(images).squeeze(dim=-1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
                train_preds.extend(preds.astype(int))
                train_labels.extend(labels.cpu().numpy().astype(int))

            acc = accuracy_score(train_labels, train_preds)
            f1 = f1_score(train_labels, train_preds)
            print(f"[Train] Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
            print(confusion_matrix(train_labels, train_preds))

            # ✅ Validation
            model.eval()
            val_loss, val_preds, val_labels = 0, [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE).float().view(-1)
                    outputs = model(images).squeeze(dim=-1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
                    val_preds.extend(preds.astype(int))
                    val_labels.extend(labels.cpu().numpy().astype(int))

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            print(f"[Val] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            print(confusion_matrix(val_labels, val_preds))

            mlflow.log_metric("train_acc", acc, step=epoch)
            mlflow.log_metric("train_f1", f1, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

        model_path = f"fracture_classifier_agegroup{age_group}.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"✅ Saved model: {model_path}")
