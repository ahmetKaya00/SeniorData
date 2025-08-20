from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import classification_report 

iris = load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

tree_model = DecisionTreeClassifier()
tree_model.fit(x_train,y_train)
tree_pred = tree_model.predict(x_test)

print(accuracy_score(y_test,tree_pred))

random_model = RandomForestClassifier()
random_model.fit(x_train,y_train)
ramdom_pred = random_model.predict(x_test)

print(accuracy_score(y_test,ramdom_pred))

new_flower = np.array([[5.1,3.5,2.3,4.5]])

prediction_tree = tree_model.predict(new_flower)
print(iris.target_names[prediction_tree[0]])
prediction_random = random_model.predict(new_flower)
print(iris.target_names[prediction_random[0]])

X_petal = x[:, [2,3]]
for label, color, species in zip([0,1,2],['red','green','blue'], iris.target_names):
    plt.scatter(X_petal[y == label, 0],
                X_petal[y == label, 1],
                c=color,label=species)
plt.xlabel("Petal Length(cm)")
plt.ylabel("Petal Width(cm)")
plt.title("Iris")
plt.legend()
plt.grid()
plt.show()


cm = confusion_matrix(y_test,tree_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Karar Agaci")
plt.show()

report = classification_report(y_test,tree_pred,target_names=iris.target_names)

print(report)