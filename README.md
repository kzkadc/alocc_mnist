# ディープラーニングによる異常検知手法ALOCC
Sabokrou, et al. "Adversarially Learned One-Class Classifier for Novelty Detection", The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 3379-3388

https://arxiv.org/abs/1802.09088

## 準備 (Chainer)
ChainerとOpenCVを使います。  
インストール：
```bash
$ sudo pip install chainer opencv-python
```

## 準備 (PyTorch)
PyTorch, PyTorch-Ignite, OpenCV, Matplotlib, scikit-learnを使います。

インストール：  
PyTorch: 公式HPを参照
```bash
$ sudo pip install pytorch-ignite opencv-python matplotlib sklearn
```


## 実行
設定ファイル`setting.json`を編集することでハイパーパラメータを設定できます。

```bash
$ python train.py 設定ファイル 出力フォルダ [-g GPUID]
```

例：  
```bash
$ python train.py setting.json result
```
