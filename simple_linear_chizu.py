# 単回帰分析

# ライブラリのインポート
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# データセットのインポート
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values  # 説明変数をXに代入
y = dataset.iloc[:, 1].values  # 目的変数をyに代入

# トレーニングセットとテストセットに分割
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# トレーニングデータを予測させる
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# テストセットの性能評価
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)
# トレーニングセットの精度を可視化
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_pred_train, color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('経験年数(年)')
plt.ylabel('給料($)')
plt.show()

# テストセットの精度を可視化
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, y_pred, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('経験年数(年)')
plt.ylabel('給料($)')
plt.show()

# 誤差を可視化
plt.scatter(y_pred, y_pred - y_test, c="blue", marker="o", label="X_test")
plt.scatter(y_pred_train, y_pred_train - y_train, c="green", marker="s", label="X_train")
plt.xlabel("予測値($)")
plt.ylabel("差($)")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=0, xmax=140000, lw=2, color='red')
plt.show()