# coding: utf-8

import numpy as np
import cv2

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers
from chainer.training import updaters, Trainer, extensions
import chainer.datasets

import matplotlib
matplotlib.use("Agg")

import model

import subprocess, pprint, json
from pathlib import Path

class GANUpdater(updaters.StandardUpdater):
	def __init__(self, iterator, gen_opt, dis_opt, l2_lam, noise_std, n_dis=1, **kwds):
		opts = {
			"gen": gen_opt,
			"dis": dis_opt
		}
		iters = {"main": iterator}
		self.n_dis = n_dis
		self.l2_lam = l2_lam
		self.noise_std = noise_std
		super().__init__(iters, opts, **kwds)
		
	def get_batch(self):
		x = self.get_iterator("main").next()
		x = np.stack(x)
		
		noise = np.random.normal(0, self.noise_std, size=x.shape).astype(np.float32)
		x_noisy = np.clip(x+noise, 0.0, 1.0) # ガウシアンノイズを付加
		
		x = Variable(x)
		x_noisy = Variable(x_noisy)
		
		if chainer.config.user_gpu_mode:
			x.to_gpu()
			x_noisy.to_gpu()
		
		return x, x_noisy
		
	def update_core(self):
		opt_gen = self.get_optimizer("gen")
		opt_dis = self.get_optimizer("dis")
		gen = opt_gen.target
		dis = opt_dis.target
		
		# update discriminator
		# 本物に対しては1，偽物に対しては0を出すように学習
		for i in range(self.n_dis):
			x, x_noisy = self.get_batch()
			x_fake = gen(x_noisy)
			
			d_real = dis(x)
			ones = dis.xp.ones(d_real.shape[0], dtype=np.int32)
			loss_d_real = F.softmax_cross_entropy(d_real, ones)
			
			d_fake = dis(x_fake)
			zeros = dis.xp.zeros(d_fake.shape[0], dtype=np.int32)
			loss_d_fake = F.softmax_cross_entropy(d_fake, zeros)
			
			loss_dis = loss_d_real + loss_d_fake
			
			dis.cleargrads()
			loss_dis.backward()
			opt_dis.update()
		
		# update generator
		# 生成した画像に対してDが1を出すようにする
		x, x_noisy = self.get_batch()
		x_fake = gen(x_noisy)
		
		d_fake = dis(x_fake)
		ones = dis.xp.ones(d_fake.shape[0], dtype=np.int32)
		loss_gen = F.softmax_cross_entropy(d_fake, ones)
		
		loss_gen_l2 = F.mean_squared_error(x, x_fake)
		
		loss_gen_total = loss_gen + self.l2_lam*loss_gen_l2
		
		gen.cleargrads()
		dis.cleargrads()
		loss_gen_total.backward()
		opt_gen.update()
		
		chainer.report({
			"generator/loss": loss_gen,
			"generator/l2": loss_gen_l2,
			"discriminator/loss": loss_dis
		})
		
		
# 生成画像を保存するextension
def ext_save_img(generator, pos_data, neg_data, out:Path, noise_std):
	try:
		out.mkdir(parents=True)
	except FileExistsError:
		pass

	@chainer.training.make_extension()
	def _ext_save_img(trainer):
		# 画像取得
		i = np.random.randint(len(pos_data))
		pos_img = pos_data[i][0]
		i = np.random.randint(len(neg_data))
		neg_img = neg_data[i][0]
		
		# ノイズ付加
		noise = np.random.normal(0, noise_std, size=pos_img.shape).astype(np.float32)
		pos_img = np.clip(pos_img+noise, 0.0, 1.0)
		noise = np.random.normal(0, noise_std, size=neg_img.shape).astype(np.float32)
		neg_img = np.clip(neg_img+noise, 0.0, 1.0)
		
		# 保存
		temp = np.squeeze(pos_img*255).astype(np.uint8)
		cv2.imwrite(str(out/"in_pos_iter_{:06d}.png".format(trainer.updater.iteration)), temp)
		temp = np.squeeze(neg_img*255).astype(np.uint8)
		cv2.imwrite(str(out/"in_neg_iter_{:06d}.png".format(trainer.updater.iteration)), temp)
		
		# shapeを調整
		pos_img = np.expand_dims(pos_img, axis=0)
		neg_img = np.expand_dims(neg_img, axis=0)
		
		pos_img = Variable(pos_img)
		neg_img = Variable(neg_img)
		if chainer.config.user_gpu_mode:
			pos_img.to_gpu()
			neg_img.to_gpu()
		
		with chainer.using_config("train", False):
			# 再構築
			pos_recon = generator(pos_img).array
			neg_recon = generator(neg_img).array
			
		# 再構築画像を保存
		if chainer.config.user_gpu_mode:
			pos_recon = generator.xp.asnumpy(pos_recon)
			neg_recon = generator.xp.asnumpy(neg_recon)
		
		pos_recon = np.squeeze(pos_recon*255).astype(np.uint8)
		neg_recon = np.squeeze(neg_recon*255).astype(np.uint8)
		
		cv2.imwrite(str(out/"out_pos_iter_{:06d}.png".format(trainer.updater.iteration)), pos_recon)
		cv2.imwrite(str(out/"out_neg_iter_{:06d}.png".format(trainer.updater.iteration)), neg_recon)

	return _ext_save_img
	

def get_mnist_num(dig_list:list, train=True) -> np.ndarray:
	"""
	指定した数字の画像だけ返す
	"""
	mnist_dataset = chainer.datasets.get_mnist(ndim=3)[0 if train else 1]	# MNISTデータ取得
	mnist_dataset = [img for img,label in mnist_dataset[:] if label in dig_list]
	mnist_dataset = np.stack(mnist_dataset)
	return mnist_dataset

def main(args):
	result_dir_path = Path(args.result_dir)
	try:
		result_dir_path.mkdir(parents=True)
	except FileExistsError:
		pass

	with Path(args.setting).open("r") as f:
		setting = json.load(f)
	pprint.pprint(setting)

	chainer.config.user_gpu_mode = (args.g >= 0)
	if chainer.config.user_gpu_mode:
		chainer.backends.cuda.get_device_from_id(args.g).use()

	# 訓練用正常データ
	mnist_neg = get_mnist_num(setting["label"]["neg"])
	
	# iteratorを作成
	kwds = {
		"batch_size": setting["iterator"]["batch_size"],
		"shuffle": True,
		"repeat": True
	}
	neg_iter = iterators.SerialIterator(mnist_neg, **kwds)
	
	generator = model.Generator()
	discriminator = model.Discriminator()
	if chainer.config.user_gpu_mode:
		generator.to_gpu()
		discriminator.to_gpu()
		
	opt_g = optimizers.Adam(**setting["optimizer"])
	opt_g.setup(generator)
	opt_d = optimizers.Adam(**setting["optimizer"])
	opt_d.setup(discriminator)
	if setting["regularization"]["weight_decay"] > 0.0:
		opt_g.add_hook(chainer.optimizer.WeightDecay(setting["regularization"]["weight_decay"]))
		opt_d.add_hook(chainer.optimizer.WeightDecay(setting["regularization"]["weight_decay"]))
	
	updater = GANUpdater(neg_iter, opt_g, opt_d, **setting["updater"])
	trainer = Trainer(updater, (setting["iteration"],"iteration"), out=args.result_dir)
	
	# テストデータを取得
	test_neg = get_mnist_num(setting["label"]["neg"], train=False)
	test_pos = get_mnist_num(setting["label"]["pos"], train=False)
	# 正常にラベル0，異常にラベル1を付与
	test_neg = chainer.datasets.TupleDataset(test_neg, np.zeros(len(test_neg), dtype=np.int32))
	test_pos = chainer.datasets.TupleDataset(test_pos, np.ones(len(test_pos), dtype=np.int32))
	test_ds = chainer.datasets.ConcatenatedDataset(test_neg, test_pos)
	test_iter = iterators.SerialIterator(test_ds, repeat=False, shuffle=True, batch_size=500)
	
	ev_target = model.EvalModel(generator, discriminator, setting["updater"]["noise_std"])
	ev_target = model.ExtendedClassifier(ev_target)
	if chainer.config.user_gpu_mode:
		ev_target.to_gpu()
	evaluator = extensions.Evaluator(test_iter, ev_target, device=args.g if chainer.config.user_gpu_mode else None)
	trainer.extend(evaluator)
	
	# 訓練経過の表示などの設定
	trigger = (5000, "iteration")
	trainer.extend(extensions.LogReport(trigger=trigger))
	trainer.extend(extensions.PrintReport(["iteration", "generator/loss", "generator/l2", "discriminator/loss"]), trigger=trigger)
	trainer.extend(extensions.ProgressBar())
	trainer.extend(extensions.PlotReport(("generator/loss","discriminator/loss"),"iteration", file_name="loss_plot.eps", trigger=trigger))
	trainer.extend(extensions.PlotReport(["generator/l2"],"iteration", file_name="gen_l2_plot.eps", trigger=trigger))
	trainer.extend(extensions.PlotReport(("validation/main/F", "validation/main/accuracy"),"iteration", file_name="acc_plot.eps", trigger=trigger))
	trainer.extend(ext_save_img(generator, test_pos, test_neg, result_dir_path/"out_images", setting["updater"]["noise_std"]), trigger=trigger)
	trainer.extend(extensions.snapshot_object(generator, "gen_iter_{.updater.iteration:06d}.model"), trigger=trigger)
	trainer.extend(extensions.snapshot_object(discriminator, "dis_iter_{.updater.iteration:06d}.model"), trigger=trigger)
	
	# 訓練開始
	trainer.run()
	
def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("setting")
	parser.add_argument("result_dir")
	parser.add_argument("-g", type=int, default=-1, help="GPU ID (negative value indicates CPU mode)")

	args = parser.parse_args()
	
	pprint.pprint(vars(args))
	main(args)
	
if __name__ == "__main__":
	parse_args()
