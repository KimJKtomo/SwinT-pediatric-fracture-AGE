import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from timm import create_model
from torch.nn import Sigmoid
from pytorch_grad_cam import HiResCAM
from cam import reshape_transform, heatmap_filter
from unified_dataset_0605 import UnifiedDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ✅ 설정값
MODEL_BASE_PATH = "0703_fracture_classifier_agegroup{}.pt"
OUTPUT_BASE_DIR = "gradcam_results"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ✅ Transform 정의
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Age 그룹 함수
def AGE_GROUP_FN(age):
    age = float(age)
    if age < 1.5:
        return 0
    elif age < 5:
        return 1
    elif age < 9:
        return 2
    elif age < 16:
        return 3
    else:
        return 4

# ✅ 전체 그룹 반복
for AGE_GROUP in [0, 1, 2, 3, 4]:
    print(f"\n🔍 Grad-CAM for Age Group {AGE_GROUP}...")
    MODEL_PATH = MODEL_BASE_PATH.format(AGE_GROUP)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"agegroup{AGE_GROUP}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ✅ 모델 로딩
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("swin_large_patch4_window12_384_in22k", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # ✅ CAM 구성
    cam = HiResCAM(
        model=model,
        target_layers=[model.layers[-1].blocks[-1].norm1],
        use_cuda=(DEVICE.type == 'cuda'),
        reshape_transform=lambda t: reshape_transform(t, height=12, width=12)
    )
    sigmoid = Sigmoid()

    # ✅ 데이터 로딩 및 필터링
    df_test = pd.read_csv("test_set_0704.csv")
    df_test = df_test[df_test["age"].astype(float).apply(AGE_GROUP_FN) == AGE_GROUP]
    df_test["label"] = df_test["fracture_visible"].apply(lambda x: 1.0 if x == 1 else 0.0)

    val_dataset = UnifiedDataset(df_test, transform=transform, task="fracture_only")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    results = []
    for i, (img, label) in enumerate(tqdm(val_loader)):
        img = img.to(DEVICE)
        with torch.no_grad():
            out = model(img).squeeze()
            prob = sigmoid(out).item()
            pred = int(prob > 0.5)

        grayscale_cam = cam(input_tensor=img)[0]
        img_path = df_test.iloc[i]["image_path"]
        orig_img = cv2.imread(img_path)
        _, _, cam_img = heatmap_filter(grayscale_cam, orig_img)
        save_name = os.path.basename(img_path).replace(".jpg", f"_pred{pred}_label{int(label.item())}.jpg")
        cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), cam_img)

        results.append({
            "filename": os.path.basename(img_path),
            "true_label": int(label.item()),
            "pred_label": pred,
            "probability": prob,
            "age_group": AGE_GROUP
        })

    # ✅ 결과 저장
    df_group = pd.DataFrame(results)
    df_group.to_csv(os.path.join(OUTPUT_DIR, "gradcam_summary.csv"), index=False)

    # ✅ Confusion Matrix 텍스트 출력 + PNG 저장
    cm = confusion_matrix(df_group["true_label"], df_group["pred_label"])
    print(f"\n📊 Confusion Matrix for Age Group {AGE_GROUP}:\n{cm}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fracture"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Fracture Confusion Matrix (Age Group {AGE_GROUP})")
    plt.tight_layout()
    cm_save_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_agegroup{AGE_GROUP}.png")
    plt.savefig(cm_save_path)
    plt.close()
