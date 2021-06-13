# Perceptron
教師あり学習。線形分類器。  
視覚と脳の機能をモデル化したものであり、パターン認識を行う。入力層と出力層のみの2層からなる。  
単純パーセプトロン (Simple perceptron) は線形分離可能な問題を有限回の反復で解くことができる一方で、線形非分離な問題を解けないことがマービン・ミンスキーとシーモア・パパートによって指摘された。  
Andゲート  
線形分離可能なデータのみ使用可能  

## 人工ニューロン
 - 人工ニューロンとは、神経細胞（ニューロン）を数学的に表現
 - ニューラルネットワークの基本的な構成要素。  
[Detail](https://cognicull.com/ja/o0hdrkf2)  

二値分類タスク
andゲート
 - -1 陰性クラス
 - 1  陽性クラス

## 実装の流れ
Pythonファイル確認    
重みw = 学習率 × （正しい値y - 予測値output） × 訓練データX    
どちらも正しければ0が帰ってくる。
標準正規分布を用いてランダム出力　要素２つに対して３つ取得しているのは、各要素ごとの重み＋バイアス  
[わかんなくなったらこれ参考](https://blog.apar.jp/deep-learning/11979/)
