import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data_classification = {
    "Age": [22,25,47,52,46,56,55,60,62,61],
    "Income": [25000,32000,47000,52000,46000,56000,55000,60000,62000,61000],
    "Puchased": [0,0,1,1,1,1,1,1,1,1],
}

df_classification = pd.DataFrame(data_classification)

x_cls = df_classification[["Age","Income"]]
y_cls = df_classification["Puchased"]
x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(x_cls,y_cls,test_size=0.2,random_state=42)

model_logistic = LogisticRegression()

model_logistic.fit(x_train_cls,y_train_cls)

y_pred = model_logistic.predict(x_test_cls)

for gerçek, tahmin in zip(y_test_cls, y_pred):
    print(f"Gerçek: {gerçek:.2f} -> Tahmin: {tahmin:.2f}")

accuracy_logistic = accuracy_score(y_test_cls,y_pred)

print(f"Doğruluk skoru: {accuracy_logistic:.4f}")