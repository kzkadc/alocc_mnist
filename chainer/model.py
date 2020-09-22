# coding: utf-8

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers, report
from chainer.training import updaters, Trainer, extensions


class Discriminator(Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            kwds = {
                "ksize": 4,
                "stride": 2,
                "pad": 1,
                "nobias": True
            }
            N_CH = 16
            self.conv1 = L.Convolution2D(1, N_CH, **kwds)		# (14,14)
            self.conv2 = L.Convolution2D(N_CH, N_CH * 2, **kwds)  # (7,7)
            self.conv3 = L.Convolution2D(
                N_CH * 2, N_CH * 4, ksize=3, stride=1, pad=1, nobias=True)  # (7,7)
            self.conv4 = L.Convolution2D(
                N_CH * 4, N_CH * 8, ksize=3, stride=1, pad=0, nobias=True)  # (5,5)
            self.conv5 = L.Convolution2D(N_CH * 8, 2, ksize=1, stride=1, pad=0)

            self.bn1 = L.BatchNormalization(N_CH, eps=1e-5)
            self.bn2 = L.BatchNormalization(N_CH * 2, eps=1e-5)
            self.bn3 = L.BatchNormalization(N_CH * 4, eps=1e-5)
            self.bn4 = L.BatchNormalization(N_CH * 8, eps=1e-5)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.leaky_relu(self.bn4(self.conv4(h)))

        h = self.conv5(h)
        h = F.mean(h, axis=(2, 3))  # global average pooling

        return h


class Generator(Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            kwds = {
                "ksize": 4,
                "stride": 2,
                "pad": 1,
                "nobias": True
            }
            N_CH = 16
            self.conv1 = L.Convolution2D(1, N_CH, **kwds)								# (14,14)
            self.conv2 = L.Convolution2D(N_CH, N_CH * 2, **kwds)							# (7,7)
            self.conv3 = L.Convolution2D(
                N_CH * 2, N_CH * 4, ksize=3, stride=1, pad=1, nobias=True)  # (7,7)
            self.conv4 = L.Convolution2D(
                N_CH * 4, N_CH * 8, ksize=3, stride=1, pad=0, nobias=True)  # (5,5)

            self.deconv4 = L.Deconvolution2D(
                N_CH * 8, N_CH * 4, ksize=3, stride=1, pad=0, nobias=True)  # (7,7)
            self.deconv3 = L.Deconvolution2D(
                N_CH * 4, N_CH * 2, ksize=3, stride=1, pad=1, nobias=True)  # (7,7)
            self.deconv2 = L.Deconvolution2D(
                N_CH * 2, N_CH, **kwds)						# (14,14)
            self.deconv1 = L.Deconvolution2D(
                N_CH, 1, ksize=4, stride=2, pad=1)		# (28,28)

            self.bn_conv1 = L.BatchNormalization(N_CH, eps=1e-5)
            self.bn_conv2 = L.BatchNormalization(N_CH * 2, eps=1e-5)
            self.bn_conv3 = L.BatchNormalization(N_CH * 4, eps=1e-5)
            self.bn_conv4 = L.BatchNormalization(N_CH * 8, eps=1e-5)
            self.bn_deconv4 = L.BatchNormalization(N_CH * 4, eps=1e-5)
            self.bn_deconv3 = L.BatchNormalization(N_CH * 2, eps=1e-5)
            self.bn_deconv2 = L.BatchNormalization(N_CH, eps=1e-5)

    def __call__(self, x):
        h = F.leaky_relu(self.bn_conv1(self.conv1(x)))
        h = F.leaky_relu(self.bn_conv2(self.conv2(h)))
        h = F.leaky_relu(self.bn_conv3(self.conv3(h)))
        h = F.leaky_relu(self.bn_conv4(self.conv4(h)))

        h = F.leaky_relu(self.bn_deconv4(self.deconv4(h)))
        h = F.leaky_relu(self.bn_deconv3(self.deconv3(h)))
        h = F.leaky_relu(self.bn_deconv2(self.deconv2(h)))
        h = F.sigmoid(self.deconv1(h))

        return h


class EvalModel(Chain):
    """
    テスト用
    GeneratorとDiscriminatorを保持して分類器のように振る舞う
    """

    def __init__(self, gen, dis, noise_std):
        super().__init__()
        with self.init_scope():
            self.gen = gen
            self.dis = dis
        self.noise_std = noise_std

    def __call__(self, x):
        noise = self.xp.random.normal(
            0, self.noise_std, size=x.shape).astype(np.float32)
        x_noise = self.xp.clip(x + noise, 0.0, 1.0)
        x_recon = self.gen(Variable(x_noise))
        y = self.dis(x_recon)

        return y


class ExtendedClassifier(L.Classifier):
    """
    Accuracyに加えてPrecision, Recall, F-measureもreportするClassifier
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.compute_accuracy = False

    def __call__(self, *args):
        loss = super().__call__(*args)
        t = args[self.label_key]

        summary = F.classification_summary(self.y, t, label_num=2)
        if chainer.config.user_gpu_mode:
            summary = [self.xp.asnumpy(v.array) for v in summary]
        else:
            summary = [v.array for v in summary]

        # NaN対策
        y = self.y.array
        pre = summary[0][1] if (y[:, 1] > y[:, 0]).sum() >= 1 else 0  # 全部負例と予測した場合は0
        rec = summary[1][1] if summary[3][1] >= 1 else 0  # そもそもバッチに負例しかないときは0
        F_measure = summary[2][1] if pre > 0 or rec > 0 else 0  # precisionとrecallが両方0のときは0
        accuracy = F.accuracy(self.y, t)
        report({
            "pre": pre,
            "rec": rec,
            "F": F_measure,
            "accuracy": accuracy
        }, self)

        return loss
