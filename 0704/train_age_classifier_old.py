# train_age_classifier.py
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
import mlflow

from SwinT_ImageOnly_Classifier import SwinTImageClassifier
from load_new_dxmodule_0620_ao import get_combined_dataset_ao

# =========================
# Args
# =========================
class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = True
    num_classes = 5
    batch_size = 8
    lr = 1e-5
    num_epochs = 40
    num_workers = 2
    patience = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "swinT_age_classifier.pt"
    run_name = "AgeGroup_Classifier"

# =========================
# Age Label Mapping
# =========================
def age_group_label(age):
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
# =========================
# Dataset Wrapper
# =========================
class AgeGroupDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(transforms.ToPILImage()(image))
        label = int(row['age_group_label'])
        return image, label

# =========================
# Evaluation
# =========================
def evaluate(model, loader, args, name="val"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds += pred.tolist()
            labels += y.numpy().tolist()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)
    print(f"\n📊 {name} Confusion Matrix:\n{cm}")
    print(classification_report(labels, preds, digits=4))
    return acc, f1, cm

# =========================
# Main
# =========================
if __name__ == '__main__':
    args = Args()
    mlflow.start_run(run_name=args.run_name)
    mlflow.log_params(vars(args))

    # 1. 전체 Kaggle 데이터
    df = get_combined_dataset_ao()
    df = df[df['source'] == 'kaggle']
    df = df[df['age'].notnull()]
    df['age'] = df['age'].astype(int)
    df['age_group_label'] = df['age'].apply(age_group_label)
    df = df[df['age_group_label'].notnull()]

    # 2. test_set.csv 제외
    df_test = pd.read_csv("test_set_0704.csv")
    test_stems = set(df_test['filestem'])
    df['filestem'] = df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df = df[~df['filestem'].isin(test_stems)].reset_index(drop=True)

    # 3. train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['age_group_label'], random_state=42)

    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_loader = DataLoader(AgeGroupDataset(train_df, tf), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AgeGroupDataset(val_df, tf), batch_size=args.batch_size, shuffle=False)

    print(f"🗂️ Train samples: {len(train_loader.dataset)}")
    print(f"🧪 Validation samples: {len(val_loader.dataset)}")

    model = SwinTImageClassifier(args).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1, counter = 0, 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss, preds, labels = 0, [], []
        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            preds += torch.argmax(out, dim=1).cpu().tolist()
            labels += y.cpu().tolist()

        tr_acc = accuracy_score(labels, preds)
        tr_f1 = f1_score(labels, preds, average='macro')
        mlflow.log_metric("train_acc", tr_acc, step=epoch)
        mlflow.log_metric("train_f1", tr_f1, step=epoch)

        val_acc, val_f1, val_cm = evaluate(model, val_loader, args, name="Val")
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), args.model_path)
            mlflow.log_artifact(args.model_path)
            print("✅ Best model saved.")
        else:
            counter += 1
            if counter >= args.patience:
                print("⏹️ Early stopping.")
                break

    mlflow.end_run()
    print("\n🎉 Age Group Classifier training completed.")
