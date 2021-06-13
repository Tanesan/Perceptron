import numpy as np
from matplotlib.colors import ListedColormap


class Perceptron(object):
    # eta: float  # 学習率　0-1
    # n_iter: int  # training data 訓練データの訓練回数
    # random_state: int  # 重みを初期化するための乱数シード

    # classに値追加
    def __init__(self, eta=0.01, n_iter=500, random_state=100):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(
            self.random_state)  # 乱数の再現性を担保　randomstateは乱数調整表のように、ランダムではあるが何回実行しても同じ値を示す。seedと同意
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # 正規分布を使った乱数調整　重みベクトルを作成
        # X.shape = [100 ,2] 100-訓練データの数, 2-特徴量の数（2要素）
        # loc Avr, scale 標準偏差　size 出力配列のサイズ
        self.errors_ = []
        for _ in range(self.n_iter):  # 訓練回数
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                # 学習率　*　target(y軸の特徴量 -1と1のデータ) - ベクトルで求めた値
                # ベクトルで求めた値と実際の統計量が同じであれば重みは正しいため、変更不要、、だから２つを引き算すると０
                # どっちがが間違ってると、重みを変更
                # ランダムで作成した重みを正と負に偏らせなければならない（＝一様ではだめ）
                # print(update * xi)
                self.w_[1:] += update * xi  # 重みを反映　重みを正or負にかたよらせる なぜ訓練データの個数、特徴量をかけるの？
                self.w_[0] += update  # 直接反映（np.dot）なぜ？
                errors += int(update != 0.0)  # 予測と離れたもの　0になればいい
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        # n.dot ベクトルの内積
        # 実際の要素Xとランダムに作成された数値の内積を取り、それにランダムで作成した値をたす。それが0以上であれば1...
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


import matplotlib.pyplot as plt
import pandas as pd
import os

s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
#
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
#
# plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(eta=0.2, n_iter=5)
ppn.fit(X, y)


#
# ppn.fit(X, y)
#
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of update')
# plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


plot_decision_regions(X, y, classifier=ppn)

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')

plt.legend(loc='upper left')
plt.show()
