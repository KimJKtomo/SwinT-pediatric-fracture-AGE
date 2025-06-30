import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from unified_dataset_0605 import UnifiedFractureDataset
from SwinT_ImageOnly_Classifier import SwinTImageClassifier


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def train_group_model(train_df, val_df, age_group, device, save_path):
    if train_df["label"].nunique() < 2:
        print(f"❌ Skipping age group {age_group} (only one class present)")
        return

    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    labels = train_df["label"].tolist()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(UnifiedFractureDataset(train_df, tf), batch_size=8, sampler=sampler)
    val_loader = DataLoader(UnifiedFractureDataset(val_df, tf), batch_size=8, shuffle=False)

    model = SwinTImageClassifier(model_name="swin_large_patch4_window12_384_in22k", pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    best_f1, patience, counter = 0, 10, 0
    for epoch in range(40):
        model.train()
        train_preds, train_labels, total_loss = [], [], 0
        for x, y in tqdm(train_loader, desc=f"[AgeGroup {age_group}] Epoch {epoch}"):
            x, y = x.to(device), y.float().to(device)
            out = model(x)
            loss = loss_fn(out, y.view(-1, 1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            train_preds += torch.sigmoid(out).cpu().detach().numpy().flatten().tolist()
            train_labels += y.cpu().numpy().tolist()

        pred_bin = [1 if p > 0.5 else 0 for p in train_preds]
        acc = accuracy_score(train_labels, pred_bin)
        f1 = f1_score(train_labels, pred_bin)
        print(f"[Train] Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                val_preds += [1 if p > 0.5 else 0 for p in prob]
                val_labels += y.numpy().tolist()

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        print(f"[Val] Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(confusion_matrix(val_labels, val_preds))

        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved: {save_path}")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping.")
                break


if __name__ == '__main__':
    df_train = pd.read_csv("age_train_tmp.csv")
    df_val = pd.read_csv("age_val_tmp.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("saved_agegroup_models", exist_ok=True)

    for group in sorted(df_train['age_group_pred'].unique()):
        print(f"\n🔹 Training for Age Group {group}...")
        train_g = df_train[df_train['age_group_pred'] == group].reset_index(drop=True)
        val_g = df_val[df_val['age_group_pred'] == group].reset_index(drop=True)
        save_path = f"saved_agegroup_models/fracture_classifier_agegroup{group}.pt"
        train_group_model(train_g, val_g, age_group=group, device=device, save_path=save_path)

    print("\n🎉 All fracture classifiers trained.")
