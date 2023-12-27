import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch
from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver

# this code is used for trainning ,setting args like below
class Args:
    img_size = 128
    num_domains = 2
    latent_dim = 16
    hidden_dim = 512
    style_dim = 64
    lambda_reg = 1
    lambda_cyc = 1
    lambda_sty = 1
    lambda_ds = 1
    ds_iter = 20000
    w_hpf = 1
    randcrop_prob = 0.5
    total_iters = 50000
    resume_iter = 30000
    batch_size = 8
    val_batch_size = 32
    lr = 1e-4
    f_lr = 1e-6
    beta1 = 0.0
    beta2 = 0.99
    weight_decay = 1e-4
    num_outs_per_domain = 10
    mode = 'train'
    num_workers = 4
    seed = 777
    train_img_dir = "Data/genkik-pro/train"
    val_img_dir = 'Data/genkik-pro/val'
    sample_dir = 'expr/samples'
    checkpoint_dir = 'expr/checkpoints'
    eval_dir = 'expr/eval'
    result_dir = 'expr/results'
    src_dir = 'assets/representative/celeba_hq/src'
    ref_dir = 'assets/representative/celeba_hq/ref'
    inp_dir = 'assets/representative/custom/female'
    out_dir = 'assets/representative/celeba_hq/src/female'
    wing_path = 'expr/checkpoints/wing.ckpt'
    lm_path = 'expr/checkpoints/celeba_lm_mean.npz'
    print_every = 10
    sample_every = 5000
    save_every = 10000
    eval_every = 1000

args = Args()
def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

solver = Solver(args)
if args.mode == 'train':
    assert len(subdirs(args.train_img_dir)) == args.num_domains
    assert len(subdirs(args.val_img_dir)) == args.num_domains
    loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                        which='source',
                                        img_size=args.img_size,
                                        batch_size=args.batch_size,
                                        prob=args.randcrop_prob,
                                        num_workers=args.num_workers),
                    ref=get_train_loader(root=args.train_img_dir,
                                        which='reference',
                                        img_size=args.img_size,
                                        batch_size=args.batch_size,
                                        prob=args.randcrop_prob,
                                        num_workers=args.num_workers),
                    val=get_test_loader(root=args.val_img_dir,
                                        img_size=args.img_size,
                                        batch_size=args.val_batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers))
    solver.train(loaders)