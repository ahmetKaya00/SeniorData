import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix

DATA_PATH = "covtype.csv"
df = pd.read_csv(DATA_PATH)

print("Veri Boyutu(Satır,Sütun)", df.shape)
print(df.head(3))

TARGET = "Cover_Type"

print(df[TARGET].value_counts().sort_index())

print(df.isna().sum().sum())

print(df.describe().iloc[:, :6])

counts = df[TARGET].value_counts().sort_index()
plt.bar(counts.index.astype(str),counts.values)
plt.title("Sinif Dagilimi")
plt.xlabel("Sinif")
plt.ylabel("Adet")
plt.show()

x = df.drop(columns=TARGET)
y = df[TARGET]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

rf_base = RandomForestClassifier(
    n_estimators = 200,
    random_state=42,
    n_jobs = 1,
    oob_score=True,
    bootstrap=True
)

print("\n>>>Baseline model eğitiliyor...")
rf_base.fit(x_train,y_train)

print("OOB Score (Baseline):", rf_base.oob_score_)
y_pred_base = rf_base.predict(x_test)

acc_base = accuracy_score(y_test,y_pred_base)
f1m_base = f1_score(y_test,y_pred_base,average="macro")

print(f"Test Accuracy: {acc_base:.4f}")
print(f"Macro-F1 Skoru: {f1m_base:.4f}")

print(classification_report(y_test,y_pred_base,digits=4))

labels_sorted = sorted(y.unique())
cm = confusion_matrix(y_test,y_pred_base,labels=labels_sorted)

print("Confusion Matrix:\n", cm)

plt.figure(figsize=(7,6))
plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
plt.title("CM-Baseline RF")
plt.colorbar()
tick_marks = np.arange(len(labels_sorted))
plt.xticks(tick_marks, labels_sorted)
plt.yticks(tick_marks, labels_sorted)
plt.xlabel("Tahmin Edilenler")
plt.ylabel("Gercek")
plt.tight_layout()
plt.show()

errors_mask = (y_test.values != y_pred_base)
total_errors = errors_mask.sum()
total = len(y_test)
print(f"\n Toplam Hata SAyısı: {errors_mask} / {total} (Oran). {total_errors/total:.2%}")

per_class_errors = dict.fromkeys(labels_sorted, 0)
for true_label, pred_label in zip(y_test.values,y_pred_base):
    if true_label != pred_label:
        per_class_errors[true_label] += 1

print("\n Sınıf bazlı hata sayıları:")
for c in labels_sorted:
    print(f"Sınıf {c}: {per_class_errors[c]} hata")

pair_counts = {}
for true_label, pred_label in zip(y_test.values, y_pred_base):
    if true_label != pred_label:
        pair_counts[(true_label,pred_label)] = pair_counts.get((true_label,pred_label),0) + 1

pair_sorted = sorted(pair_counts.items(), key=lambda kv:kv[1],reverse=True)[:10]

print("\n En çok karşılaşılan 10 çift:")
for (true,pred), cnt in pair_sorted:
    print(f"{true} -> {pred}: {cnt} kez")


rf_bal = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=1,
    oob_score=True,
    class_weight="balanced",
    bootstrap=True
)

print("\n>>> Balaced RF Eğitiliyor")
rf_bal.fit(x_train,y_train)

print("OOB Score", rf_bal.oob_score_)

y_pred_bal = rf_bal.predict(x_test)

acc_bal = accuracy_score(y_test,y_pred_bal)
f1m_bal = f1_score(y_test,y_pred_bal,average="macro")

print(f"Test Accuracy: {acc_bal:.4f}")
print(f"Macro-F1 Skoru: {f1m_bal:.4f}")

print(classification_report(y_test,y_pred_bal,digits=4))

