import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from UNet_class_model import UNet3DClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import os
import pandas as pd


def get_performance(y_true, y_pred, y_pred_probs):
    # AUROC 계산을 위해 확률처럼 간주 (이진값 사용 시 ROC AUC는 동작은 하지만 계단식으로 나타남)
    #y_pred_probs = y_pred

    # 평가 지표 계산
    auroc = roc_auc_score(y_true, y_pred_probs)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    precision = precision_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    # 평균 ± 표준편차 형식으로 출력 (지금은 값 하나이므로 SD는 0)
    results = {
        "AUROC": (auroc, 0),
        "Accuracy": (accuracy, 0),
        "Sensitivity": (sensitivity, 0),
        "Specificity": (specificity, 0),
        "Precision": (precision, 0),
        "F1 Score": (f1, 0),
    }
    return results


def compute_metrics_with_ci(y_true, y_pred, y_score, n_bootstrap=1000, ci=95, random_state=42):
    rng = np.random.RandomState(random_state)
    metrics = {
        "AUROC": [],
        "Accuracy": [],
        "Sensitivity": [],
        "Specificity": [],
        "Precision": [],
        "F1-score": []
    }

    n = len(y_true)
    for _ in range(n_bootstrap):
        # bootstrap 샘플링
        idx = rng.choice(np.arange(n), size=n, replace=True)
        y_true_bs = y_true[idx]
        y_pred_bs = y_pred[idx]
        y_score_bs = y_score[idx]

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_bs, y_pred_bs).ravel()

        # 각 지표 계산
        try:
            metrics["AUROC"].append(roc_auc_score(y_true_bs, y_score_bs))
        except:
            pass
        metrics["Accuracy"].append(accuracy_score(y_true_bs, y_pred_bs))
        metrics["Sensitivity"].append(tp / (tp + fn) if (tp+fn)>0 else np.nan)
        metrics["Specificity"].append(tn / (tn + fp) if (tn+fp)>0 else np.nan)
        metrics["Precision"].append(precision_score(y_true_bs, y_pred_bs, zero_division=0))
        metrics["F1-score"].append(f1_score(y_true_bs, y_pred_bs, zero_division=0))

    # 평균, 표준편차, CI 계산 (소수점 4자리 반올림)
    results = {}
    alpha = (100 - ci) / 2
    for key, values in metrics.items():
        values = np.array(values)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        lower = np.nanpercentile(values, alpha)
        upper = np.nanpercentile(values, 100 - alpha)
        results[key] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            f"{ci}% CI": (round(lower, 4), round(upper, 4))
        }

    return results

def compute_metrics_with_ci_self(y_true, y_pred, y_score, n_bootstrap=1000, ci=95, random_state=42):
    rng = np.random.RandomState(random_state)
    metrics = {
        "AUROC": [],
        "Accuracy": [],
        "Sensitivity": [],
        "Specificity": [],
        "Precision": [],
        "F1-score": []
    }

    n = len(y_true)


    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 각 지표 계산
    try:
        metrics["AUROC"].append(roc_auc_score(y_true, y_score))
    except:
        pass
    metrics["Accuracy"].append(accuracy_score(y_true, y_pred))
    metrics["Sensitivity"].append(tp / (tp + fn) if (tp+fn)>0 else np.nan)
    metrics["Specificity"].append(tn / (tn + fp) if (tn+fp)>0 else np.nan)
    metrics["Precision"].append(precision_score(y_true, y_pred, zero_division=0))
    metrics["F1-score"].append(f1_score(y_true, y_pred, zero_division=0))

    # 평균, 표준편차, CI 계산 (소수점 4자리 반올림)
    results = {}
    alpha = (100 - ci) / 2
    for key, values in metrics.items():
        values = np.array(values)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        lower = np.nanpercentile(values, alpha)
        upper = np.nanpercentile(values, 100 - alpha)
        results[key] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            f"{ci}% CI": (round(lower, 4), round(upper, 4))
        }

    return results


def drow_AUROC_chart(y_true, y_pred, y_score, save_name):
    # ROC curve를 위한 fpr 기준선
    fpr_mean = np.linspace(0, 1, 100)
    tpr_list = []
    auroc_list = []

    # 부트스트랩 반복
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        # 데이터 리샘플링
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue  # 클래스가 하나만 있으면 스킵
        fpr, tpr, _ = roc_curve(y_true[indices], y_score[indices])
        roc_auc = auc(fpr, tpr)
        auroc_list.append(roc_auc)

        # 보간해서 동일한 fpr 기준으로 맞추기
        tpr_interp = np.interp(fpr_mean, fpr, tpr)
        tpr_list.append(tpr_interp)

    # 평균 및 표준편차 계산
    tpr_mean = np.mean(tpr_list, axis=0)
    tpr_std = np.std(tpr_list, axis=0)
    auroc_mean = np.mean(auroc_list)
    auroc_std = np.std(auroc_list)
    print(f"auroc_mean: {auroc_mean}   ")

    # 신뢰구간 (95%)
    ci_lower = np.percentile(auroc_list, 2.5)
    ci_upper = np.percentile(auroc_list, 97.5)

    # 그림 그리기
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_mean, tpr_mean, color="blue",
             label=f"Model (AUROC = {auroc_mean:.3f} ± {auroc_std:.3f})")
    plt.fill_between(fpr_mean, tpr_mean - tpr_std, tpr_mean + tpr_std,
                     color="blue", alpha=0.2, label="± 1 std. dev.")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Chance")

    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curve with Confidence Interval")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_name)
    #plt.show()

    return 0

model_name = "UNet_encoder"
save_result = "test_all_3_hospital.xlsx"
data_dir = r'../Test_data/test_all_3_hospital'
model_path = f'../weights/{model_name}_lab_data.pth'
result_dir = f'../results/{model_name}'
save_name = os.path.join(result_dir, f"{model_name}_AUROC.jpg")
os.makedirs(result_dir, exist_ok=True)
result_xlsx = os.path.join(result_dir, save_result)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 모델 로딩
model = UNet3DClassifier(in_channels=1, hidden_dim=256, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

def preprocess(arr):

    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    arr = arr.astype(np.float32)
    arr = arr / (arr.max() if arr.max() > 0 else 1.0)
    return arr

results = []

with torch.no_grad():
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith('.pkl'):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'rb') as f:
            d = pickle.load(f)
        arr = d['data']
        label = d['label']
        true_label = 0 if label == 'N' else 1
        arr = preprocess(arr)
        x = torch.from_numpy(arr).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_label = logits.argmax(dim=1).item()
        prob_F = probs[0, 1].item()
        prob_N = probs[0, 0].item()
        prob = probs[0, pred_label].item()

        data_aug = d.get('data_augmentation')
        if fname.endswith("_Y_0.pkl"):
            data_aug = "N"
        side = d.get('side')

        results.append([fname, side, data_aug, pred_label, true_label, round(prob_F, 4), round(prob_N, 4), round(prob, 4)])

df = pd.DataFrame(
    results,
    columns=['ID', 'Side', 'Data Augmentation', 'Pred Label', 'True Label', 'P(F)', 'P(N)', 'P']
)
df.to_excel(result_xlsx, index=False)

##  확장 데이터 제거
df = df[df['Data Augmentation'] == 'N']
df = df.reset_index(drop=True)

y_true = df['True Label'].values
y_pred = df['Pred Label'].values
y_score = df['P(F)'].values

drow_AUROC_chart(y_true, y_pred, y_score, save_name)

#tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
rst = compute_metrics_with_ci(y_true, y_pred, y_score, n_bootstrap=1000, ci=95, random_state=42)
#rst = compute_metrics_with_ci_self(y_true, y_pred, y_score, n_bootstrap=1000, ci=95, random_state=42)  ## n_bootstrap 제거

print(f"모델 성능: {result_dir}")
print(rst)

with open(f"{os.path.join(result_dir, model_name)}_performance.txt", "w") as f:
    for metric, values in rst.items():
        line = f"{metric}: mean={values['mean']}, std={values['std']}, 95% CI={values['95% CI']}\n"
        f.write(line)

print("✅ 'results.txt' 파일 저장 완료")



### 준희 원본 코드
# preds = df['Pred Label'].values
# trues = df['True Label'].values
#
# precision_0 = precision_score(trues, preds, pos_label=0)
# precision_1 = precision_score(trues, preds, pos_label=1)
# recall_0 = recall_score(trues, preds, pos_label=0)
# recall_1 = recall_score(trues, preds, pos_label=1)
# f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
# f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
# acc = accuracy_score(trues, preds)
#
# print(f"N: Precision={precision_0:.4f}, Recall={recall_0:.4f}, F1={f1_0:.4f}")
# print(f"F: Precision={precision_1:.4f}, Recall={recall_1:.4f}, F1={f1_1:.4f}")
# print(f"Accuracy: {acc:.4f}")




