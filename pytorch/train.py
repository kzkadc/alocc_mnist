import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from ignite.engine.engine import Engine, Events

import numpy as np
import cv2
from sklearn import metrics
import matplotlib.pyplot as plt

import pprint
from pathlib import Path
import json

from model import get_discriminator, get_generator, Detector


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("setting")
    parser.add_argument("result_dir")
    parser.add_argument("-g", type=int, default=-1,
                        help="GPU ID (negative value indicates CPU mode)")

    args = parser.parse_args()

    pprint.pprint(vars(args))
    main(args)


def main(args):
    result_dir_path = Path(args.result_dir)
    try:
        result_dir_path.mkdir(parents=True)
    except FileExistsError:
        pass

    with Path(args.setting).open("r") as f:
        setting = json.load(f)
    pprint.pprint(setting)

    if args.g >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.g:d}")
        print(f"GPU mode: {args.g:d}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    mnist_neg = get_mnist_num(set(setting["label"]["neg"]))
    neg_loader = DataLoader(
        mnist_neg, batch_size=setting["iterator"]["batch_size"])

    generator = get_generator().to(device)
    discriminator = get_discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(),
                             lr=setting["optimizer"]["alpha"],
                             betas=(setting["optimizer"]["beta1"], setting["optimizer"]["beta2"]),
                             weight_decay=setting["regularization"]["weight_decay"])
    opt_d = torch.optim.Adam(discriminator.parameters(),
                             lr=setting["optimizer"]["alpha"],
                             betas=(setting["optimizer"]["beta1"], setting["optimizer"]["beta2"]),
                             weight_decay=setting["regularization"]["weight_decay"])

    trainer = Engine(GANTrainer(generator, discriminator, opt_g, opt_d, **setting["updater"]))

    # テスト用
    test_neg = get_mnist_num(set(setting["label"]["neg"]), train=False)
    test_neg_loader = DataLoader(test_neg, setting["iterator"]["batch_size"])
    test_pos = get_mnist_num(set(setting["label"]["pos"]), train=False)
    test_pos_loader = DataLoader(test_pos, setting["iterator"]["batch_size"])
    detector = Detector(generator, discriminator, setting["updater"]["noise_std"], device).to(device)

    log_dict = {}
    evaluator = evaluate_accuracy(log_dict, detector, test_neg_loader, test_pos_loader, device)
    plotter = plot_metrics(log_dict, ["accuracy", "precision", "recall", "f"], "iteration",
                           result_dir_path / "metrics.pdf")
    printer = print_logs(log_dict, ["iteration", "accuracy", "precision", "recall", "f"])
    img_saver = save_img(generator, test_pos, test_neg, result_dir_path / "images",
                         setting["updater"]["noise_std"], device)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), evaluator)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), plotter)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), printer)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), img_saver)

    trainer.run(neg_loader, max_epochs=setting["iteration"] // setting["iterator"]["batch_size"])


def get_mnist_num(dig_set: set, train=True):
    # 指定した数字の画像だけ返す
    mnist = MNIST(root=".", train=train, download=True,
                  transform=lambda x: np.asarray(x, dtype=np.float32).reshape(1, 28, 28) / 255)
    indices = [i for i, (_, label) in enumerate(mnist) if label in dig_set]
    return Subset(mnist, indices)


class GANTrainer:
    def __init__(self, gen: nn.Module, dis: nn.Module,
                 opt_gen, opt_dis, l2_lam: float, noise_std: float, n_dis: int = 1, device="cpu"):
        self.gen = gen
        self.dis = dis
        self.opt_gen = opt_gen
        self.opt_dis = opt_dis
        self.l2_lam = l2_lam
        self.noise_std = noise_std
        self.n_dis = 1
        self.device = device

    def prepare_batch(self, batch):
        x, _ = batch
        noise = np.random.normal(0, self.noise_std, size=x.shape)
        noise = torch.from_numpy(noise).float()
        x_noisy = (x + noise).clamp(0.0, 1.0)  # ガウシアンノイズを付加

        return x.to(self.device), x_noisy.to(self.device)

    def __call__(self, engine, batch):
        x, x_noisy = self.prepare_batch(batch)
        batch_size = len(x) // (self.n_dis + 1)

        self.gen.train()
        self.dis.train()

        # update discriminator
        # 本物に対しては1，偽物に対しては0を出すように学習
        for i in range(self.n_dis):
            self.opt_dis.zero_grad()
            self.opt_gen.zero_grad()

            x_ = x[i * batch_size:(i + 1) * batch_size]
            x_noisy_ = x_noisy[i * batch_size:(i + 1) * batch_size]
            x_fake = self.gen(x_noisy_).detach()

            loss_dis = self.discriminator_loss(x_fake, x_)

            loss_dis.backward()
            self.opt_dis.step()

        # update generator
        # 生成した画像に対してDが1を出すようにする
        x_ = x[self.n_dis * batch_size:(self.n_dis + 1) * batch_size]
        x_noisy_ = x_noisy[self.n_dis * batch_size:(self.n_dis + 1) * batch_size]

        self.opt_gen.zero_grad()
        self.opt_dis.zero_grad()

        x_fake = self.gen(x_noisy_)
        loss_dict = self.generator_loss(x_fake, x_)

        loss_dict["loss_gen_total"].backward()
        self.opt_gen.step()

        return {
            "dis_loss": loss_dis.item(),
            "gen_loss": loss_dict["loss_gen"].item(),
            "gen_loss_l2": loss_dict["loss_gen_l2"].item()
        }

    def discriminator_loss(self, x_fake, x_real):
        d_real = self.dis(x_real)
        ones = torch.ones(len(d_real), dtype=torch.long, device=self.device)
        loss_d_real = F.cross_entropy(d_real, ones)

        d_fake = self.dis(x_fake)
        zeros = torch.zeros(len(d_fake), dtype=torch.long, device=self.device)
        loss_d_fake = F.cross_entropy(d_fake, zeros)

        return loss_d_real + loss_d_fake

    def generator_loss(self, x_fake, x_real) -> dict:
        d_fake = self.dis(x_fake)
        ones = torch.ones(len(d_fake), dtype=torch.long, device=self.device)
        loss_gen = F.cross_entropy(d_fake, ones)

        loss_gen_l2 = F.mse_loss(x_real, x_fake)
        loss_gen_total = loss_gen + self.l2_lam * loss_gen_l2
        return {
            "loss_gen": loss_gen,
            "loss_gen_l2": loss_gen_l2,
            "loss_gen_total": loss_gen_total
        }


# 生成画像を保存
def save_img(generator: nn.Module, pos_data, neg_data, out: Path, noise_std: float, device):
    try:
        out.mkdir(parents=True)
    except FileExistsError:
        pass

    def _save_img(engine):
        generator.eval()

        # 画像取得
        i = np.random.randint(len(pos_data))
        pos_img, _ = pos_data[i]
        i = np.random.randint(len(neg_data))
        neg_img, _ = neg_data[i]

        # ノイズ付加
        noise = np.random.normal(0, noise_std, size=pos_img.shape)
        pos_img = np.clip(pos_img + noise, 0.0, 1.0)
        noise = np.random.normal(0, noise_std, size=neg_img.shape)
        neg_img = np.clip(neg_img + noise, 0.0, 1.0)

        # 保存
        temp = (pos_img * 255).astype(np.uint8).squeeze()
        cv2.imwrite(str(out / f"in_pos_iter_{engine.state.iteration:06d}.png"), temp)
        temp = (neg_img * 255).astype(np.uint8).squeeze()
        cv2.imwrite(str(out / f"in_neg_iter_{engine.state.iteration:06d}.png"), temp)

        # shapeを調整
        pos_img = torch.from_numpy(pos_img).float().unsqueeze(0).to(device)
        neg_img = torch.from_numpy(neg_img).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # 再構築
            pos_recon = generator(pos_img).detach().cpu().numpy() * 255
            neg_recon = generator(neg_img).detach().cpu().numpy() * 255

        pos_recon = pos_recon.squeeze().astype(np.uint8)
        neg_recon = neg_recon.squeeze().astype(np.uint8)
        cv2.imwrite(str(out / f"out_pos_iter_{engine.state.iteration:06d}.png"), pos_recon)
        cv2.imwrite(str(out / f"out_neg_iter_{engine.state.iteration:06d}.png"), neg_recon)

    return _save_img


def evaluate_accuracy(log_dict: dict, detector: nn.Module, neg_loader, pos_loader, device):
    # テストデータに対して精度を評価
    def _evaluate(engine):
        detector.eval()
        neg_preds = []
        pos_preds = []
        labels = []
        with torch.no_grad():
            for batch in neg_loader:
                y = _inference(batch)
                neg_preds.append(y)

            neg_preds = torch.cat(neg_preds).detach().cpu().numpy()
            labels.append(np.zeros(len(neg_preds), dtype=np.int32))

            for batch in pos_loader:
                y = _inference(batch)
                pos_preds.append(y)

            pos_preds = torch.cat(pos_preds).detach().cpu().numpy()
            labels.append(np.ones(len(pos_preds), dtype=np.int32))

        labels = np.concatenate(labels)
        preds = np.concatenate([neg_preds, pos_preds])

        accuracy = metrics.accuracy_score(labels, preds)
        precision, recall, f, _ = metrics.precision_recall_fscore_support(labels, preds, zero_division=0)

        _append_metrics("iteration", engine.state.iteration)
        _append_metrics("accuracy", accuracy)
        _append_metrics("precision", precision[1])
        _append_metrics("recall", recall[1])
        _append_metrics("f", f[1])

    def _append_metrics(key, value):
        if key in log_dict:
            log_dict[key].append(value)
        else:
            log_dict[key] = [value]

    def _inference(batch):
        x, _ = batch
        x = x.to(device)
        y = detector(x).argmax(dim=1)
        return y

    return _evaluate


def plot_metrics(log_dict: dict, y_keys, x_key, out_path: Path):
    def _plot(engine):
        plt.figure()
        for k in y_keys:
            plt.plot(log_dict[x_key], log_dict[k], label=k)

        plt.legend()
        plt.savefig(str(out_path))
        plt.close()

    return _plot


def print_logs(log_dict: dict, keys):
    def _print(engine):
        outs = []
        for k in keys:
            outs.append(f"{k}: {log_dict[k][-1]}")

        print(", ".join(outs))

    return _print


if __name__ == "__main__":
    parse_args()
