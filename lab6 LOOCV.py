import pandas as pd
from sklearn.linear_model import LinearRegression

file_path = '/content/sample_data/Iris.csv'
data = pd.read_csv(file_path)
print(data.head())
X=data.iloc[:, :-1].values 
y=data.iloc[:, -1].values

classes = list(set(y)) 

class_to_num = {cls: idx for idx, cls in enumerate(classes)}

y_num = [class_to_num[label] for label in y]

correct = 0
n = len(X)
for i in range(n):
  X_train = [] 
  y_train_num = []  
  for j in range(n):
    if j != i:
      X_train.append(X[j])
      y_train_num.append(y_num[j])

    X_test = [X[i]]
    y_test_num = y_num[i]

model = LinearRegression()
model.fit(X_train, y_train_num)

y_pred_continuous = model.predict(X_test)[0]

predicted_class_index = round(y_pred_continuous)
predicted_class_index = max(0, min(predicted_class_index, len(classes) - 1))

if predicted_class_index == y_test_num:
  correct += 1
  print(predicted_class_index, y_test_num)
else:
  print(predicted_class_index, y_test_num)


accuracy = (correct / n) * 100 
print(f"LOOCV Accuracy with Linear Regression: {accuracy:.4f}","%")
