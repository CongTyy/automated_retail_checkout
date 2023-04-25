# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Train a model on a dataset

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco128.yaml imgsz=640 epochs=100 batch=16
"""
import os
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm

from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)

# GAN
from ultralytics.nn.gan import *
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torch.autograd import grad
from visualize import visualize_feature


class BaseTrainer:
    """
    BaseTrainer

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        check_resume (method): Method to check if training should be resumed from a saved checkpoint.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to last checkpoint.
        best (Path): Path to best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.validator = None
        self.model = None
        self.metrics = None
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in (-1, 0) else True))
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataloaders.
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.endswith('.yaml') or self.args.task in ('detect', 'segment'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{self.args.data}' error ❌ {e}")) from e

        self.trainset, self.testset, self.realset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """
        Appends the given callback.
        """
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """
        Overrides the existing callbacks with the given callback.
        """
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        # Allow device='', device=None on Multi-GPU systems to default to device=0
        if isinstance(self.args.device, int) or self.args.device:  # i.e. device=0 or device=[0,1,2,3]
            world_size = torch.cuda.device_count()
        elif torch.cuda.is_available():  # i.e. device=None or device=''
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            cmd, file = generate_ddp_command(world_size, self)  # security vulnerability in Snyk scans
            try:
                LOGGER.info(f'Running DDP command {cmd}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(RANK, world_size)

    def _setup_ddp(self, rank, world_size):
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '9020'
        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', rank)
        LOGGER.info(f'DDP settings: RANK {rank}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo', rank=rank, world_size=world_size)

    def _setup_train(self, rank, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        # Check AMP
        self.amp = torch.tensor(True).to(self.device)
        if RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
        if self.batch_size == -1:
            if RANK == -1:  # single-GPU only, estimate best batch size
                self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # dataloaders
        batch_size = self.batch_size // world_size if world_size > 1 else self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=rank, mode='train')
        self.real_loader = self.get_dataloader(self.realset, batch_size=batch_size, rank=rank, mode="train")
        if rank in (-1, 0):
            self.val_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.test_loader = self.get_dataloader(self.realset, batch_size=batch_size * 2, rank=-1, mode='val')

            self.validator = self.get_validator(self.val_loader)
            self.test_real = self.get_validator(self.test_loader)
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
            if self.args.plots and not self.args.v5loader:
                self.plot_training_labels()
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')


        #------------------------------------------------------------------------------------------------GAN----------------------------------------------------------------------------------#
        # self.c_lr = 1e-4
        # self.real_pbar = enumerate(self.real_loader)
        # self.domain_model = Domain(256).to(self.device)
        # self.domain_optim = torch.optim.Adam(self.domain_model.parameters(), lr = self.c_lr, betas=(0.5, 0.999))
        # self.combine_res = 40
        # self.domain_loss = nn.NLLLoss().to(self.device)
        # self.writer = SummaryWriter(log_dir="./runs/gan", )
        # self.step_gan = 0
        # self.c_iters = 1
        # self.weight_cliping_limit = 0.01
        
        # self.c_lr = 1e-4
        # self.real_pbar = enumerate(self.real_loader)
        # self.critic = Critic(64).to(self.device)
        # self.critic.apply(self.critic.init_weights)
        # self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = self.c_lr, betas=(0.5, 0.999))
        # self.combine_res = 40
        # self.writer = SummaryWriter(log_dir=self.save_dir / "gan")
        # self.step_gan = 0
        # self.weight_cliping_limit = 0.01

        self.d_lr = 1e-4
        self.real_pbar = enumerate(self.real_loader)
        self.writer = SummaryWriter(log_dir=self.save_dir / "gan")
        self.step_gan = 0
        self.weight_cliping_limit = 2.0
        # self.dcgan_cri = nn.CrossEntropyLoss()
        self.dcgan_cri = nn.BCEWithLogitsLoss()

        # -------------#
        self.d1 = D_layer4().to(self.device)
        # self.d1.apply(weights_init)
        self.d1_optim = torch.optim.Adam(self.d1.parameters(), lr = self.d_lr, betas=(0.5, 0.9))

        self.d2 = D_layer6().to(self.device)
        # self.d2.apply(weights_init)
        self.d2_optim = torch.optim.Adam(self.d2.parameters(), lr = self.d_lr, betas=(0.5, 0.9))

        self.d3 = D_layer9().to(self.device)
        # self.d3.apply(weights_init)
        # self.d3.load_state_dict(torch.load("yolov8/one_layer_dcGAN/train4/d3.pth"))
        self.d3_optim = torch.optim.Adam(self.d3.parameters(), lr = self.d_lr, betas=(0.5, 0.9))
        # ------------ #


    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)

        self._setup_train(rank, world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if rank != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

            if rank in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                rgl = False
                wgan = False
                wgan_domain = True
                self.dcgan = False
                tsne = False
                self._onelayerdcgan = False
                # torch.use_deterministic_algorithms(True, warn_only=True)
                with torch.cuda.amp.autocast(self.amp):
                    if rgl or wgan or wgan_domain or self.dcgan or tsne or self._onelayerdcgan:
                        try:
                            batch_real = next(iter(self.real_pbar))[1]
                            # batch_real = self.real_pbar.__next__()[1]
                            # if len(batch_real['img']) != self.batch_size:
                            #     self.real_pbar = enumerate(self.real_loader)
                            #     batch_real = self.real_pbar.__next__()[1]
                        except StopIteration:
                            self.real_pbar = enumerate(self.real_loader)
                            batch_real = next(iter(self.real_pbar))[1]
                        batch_real = self.preprocess_batch(batch_real)

                    batch = self.preprocess_batch(batch)

                    # if tsne:
                    #     self.model.eval()
                    #     _yfs = []
                    #     with torch.no_grad():
                    #         for i, batch in pbar:
                    #             batch = self.preprocess_batch(batch)
                    #             yfsource = self.model(batch['img'], gan = True)
                    #             yfs = yfsource[0].reshape(1, -1).cpu().numpy() # 8 x D
                    #             for yf in yfs:
                    #                 _yfs.append(yf)
                    #         np.save('f1.npy', _yfs)
                    #             # yftarget = self.model(batch_real['img'], gan = True)

                    #     exit()

                    if rgl:
                        self.domain_optim.zero_grad()
                        # source 
                        preds, yfs = self.model(batch['img'], gan = True)
                        # yf = []
                        # for f in yfs: # [16 64 160 160] [16 128 80 80] [16 256 40 40]  
                        #     kernel_size = f.size(3)//self.combine_res
                        #     yf.append(integrating(f, kernel_size))
                        # yf = torch.concat(yf, dim = 1)
                        yf = yfs[-1]
                        domain_label = Variable(torch.zeros(batch['img'].size(0), device=self.device).long())
                        domain_output = self.domain_model(yf)
                        domain_s_loss = self.domain_loss(domain_output.view(-1, 2), domain_label)
                        self.loss, self.loss_items = self.criterion(preds, batch)

                        # target
                        _, yft = self.model(batch_real['img'], gan = True)
                        # yf = []
                        # for f in yft: # [16 64 160 160] [16 128 80 80] [16 256 40 40]  
                        #     kernel_size = f.size(3)//self.combine_res
                        #     yf.append(integrating(f, kernel_size))
                        # yf = torch.concat(yf, dim = 1)
                        yf = yft[-1]
                        domain_label = Variable(torch.ones(batch_real['img'].size(0), device=self.device).long())
                        domain_output = self.domain_model(yf)
                        domain_t_loss = self.domain_loss(domain_output.view(-1, 2), domain_label)
                        self.writer.add_scalar('domain_s_loss', domain_s_loss.item(), self.step_gan)
                        self.writer.add_scalar('domain_t_loss', domain_t_loss.item(), self.step_gan)

                    elif wgan:
                        yff = self.model(batch['img'], gan = True)[0]
                        yfr = self.model(batch_real['img'], gan = True)[0]
                        # train Critic

                        self.critic_optim.zero_grad()
                        c_fake = torch.mean(self.critic(yff.detach()))
                        c_real = torch.mean(self.critic(yfr.detach()))
                        gp = self.calc_gradient_penalty(self.critic, yfr, yff, self.device, LAMBDA = 10)
                        c_loss = c_fake - c_real + gp
                        c_loss.backward()
                        self.critic_optim.step()
                        
                        self.writer.add_scalar('c_fake', c_fake.item(), self.step_gan)
                        self.writer.add_scalar('c_real', -c_real.item(), self.step_gan)
                        self.writer.add_scalar('c_loss', c_loss.item(), self.step_gan)
                        self.writer.add_scalar('gp', gp.item(), self.step_gan)
                        
                        if (self.step_gan + 1) % 5 != 0:
                            self.step_gan += 1
                            continue 
                            

                        # train YOLO
                        preds, yff = self.model(batch['img'], gan = True, full = True)
                        self.loss, self.loss_items = self.criterion(preds, batch)
                        c_fake = -torch.mean(self.critic(yff[0]))
                        scale = abs(c_fake.item())/self.loss.item() if abs(c_fake.item()) > self.loss.item() else self.loss.item()/abs(c_fake.item())
                        self.loss = 5*self.loss + c_fake
                        self.writer.add_scalar('scale', scale, self.step_gan)
                        self.writer.add_scalar('cg_fake', c_fake.item(), self.step_gan)

                    elif wgan_domain:

                        # self.set_requires_grad(self.d1, True)
                        # self.set_requires_grad(self.d2, True)
                        self.set_requires_grad(self.d3, True)

                        self.model.eval()
                        with torch.no_grad():
                            yfsource = self.model(batch['img'], gan = True)
                            yftarget = self.model(batch_real['img'], gan = True)

                        k_critic = 5
                        for _ in range(k_critic):
                            # 128 20 20 --> B,51200
                            yfsource_C = self.d3.encode(yfsource[2]).view(yfsource[2].shape[0], -1)  
                            yftarget_C = self.d3.encode(yftarget[2]).view(yftarget[2].shape[0], -1)

                            gp = self.gradient_penalty(self.d3, yfsource_C, yftarget_C, LAMBDA = 10)
                            critic_s = self.d3.last(yfsource_C)
                            critic_t = self.d3.last(yftarget_C)
                            wasserstein_distance = critic_s.mean() - critic_t.mean()
                            critic_cost = -wasserstein_distance + gp

                            self.d3_optim.zero_grad()
                            critic_cost.backward()
                            self.d3_optim.step()

                    
                            self.writer.add_scalar('critic_loss/critic_s', critic_s.mean().item(), self.step_gan) 
                            self.writer.add_scalar('critic_loss/critic_t', critic_t.mean().item(), self.step_gan) 
                            self.writer.add_scalar('critic_loss/gp3', gp.item(), self.step_gan) 
                            self.writer.add_scalar('critic_loss/critic_cost', critic_cost.item(), self.step_gan)
                            self.step_gan += 1
                  
                        
                        # train YOLO
                        self.set_requires_grad(self.d3, False)
                        k_yolo = 10
                        alpha = 0.1
                        for _ in range(k_yolo):
                            preds, yfsource = self.model(batch['img'], full = True, gan = True)
                            yftarget = self.model(batch_real['img'], gan = True)

                            yfsource_G = self.d3.encode(yfsource[2]).view(yfsource[2].shape[0], -1)  
                            yftarget_G = self.d3.encode(yftarget[2]).view(yftarget[2].shape[0], -1)

                            self.loss, self.loss_items = self.criterion(preds, batch)
                            wasserstein_distance = self.d3.last(yfsource_G).mean() - self.d3.last(yftarget_G).mean()


                            self.writer.add_scalar('domain/yolo_loss', self.loss.item(), self.step_gan)
                            self.writer.add_scalar('domain/wasserstein_distance', wasserstein_distance.item(), self.step_gan)
                            self.step_gan += 1

                            self.scaler.scale(self.loss + alpha*wasserstein_distance).backward()
                            self.optimizer_step()


                    elif self._onelayerdcgan:
                        '''
                        4: 64 160 160
                        6: 128 80 80
                        9: 256 40 40 
                        '''
                        # self.set_requires_grad(self.model, False)
                        # # if epoch % 1 == 0:
                        self.set_requires_grad(self.d1, True)
                        self.set_requires_grad(self.d2, True)
                        self.set_requires_grad(self.d3, True)

                        self.model.eval()
                        with torch.no_grad():
                            yfsource = self.model(batch['img'], gan = True)
                            yftarget = self.model(batch_real['img'], gan = True)
                            
                        # visualize_feature(yfsource, "yfsource.png", batch['img'])
                        # visualize_feature(yftarget, "yftarget.png", batch_real['img'])
                        # train Critic
                        self.d3_optim.zero_grad()
                        target_gt = Variable(torch.zeros(self.batch_size, 1)).to(self.device)
                        source_gt = Variable(torch.ones(self.batch_size, 1)).to(self.device)

                        # ds_3 = torch.sigmoid(self.d3(yfsource[2].detach()))
                        # ds_3_loss = 0.5*(torch.mean((1 - ds_3)**2)) 
                        ds_3 = self.d3(yfsource[2].detach())
                        # ds_3 = torch.log_softmax(ds_3, 1)
                        ds_3_loss = self.dcgan_cri(ds_3, source_gt)

                        # dt_3 = torch.sigmoid(self.d3(yftarget[2].detach()))
                        # dt_3_loss = 0.5*(torch.mean((dt_3)**2)) # Weighted Least-squares (LS) Loss
                        dt_3 = self.d3(yftarget[2].detach())
                        # dt_3 = torch.log_softmax(dt_3, 1)
                        dt_3_loss = self.dcgan_cri(dt_3, target_gt)

                        c_loss = ds_3_loss + dt_3_loss
                        c_loss.backward()
                        self.d3_optim.step()

                        self.writer.add_scalar('loss_d/ds_3_loss', ds_3_loss.item(), self.step_gan) 
                        self.writer.add_scalar('loss_d/dt_3_loss', dt_3_loss.item(), self.step_gan) 
                        self.writer.add_scalar('loss_d/dloss', c_loss.item(), self.step_gan)
                        self.writer.add_scalar('pred_d/pred_dt_3', torch.sigmoid(dt_3).mean().item(), self.step_gan) 
                        self.writer.add_scalar('pred_d/pred_ds_3', torch.sigmoid(ds_3).mean().item(), self.step_gan) 
                        
                        # if (self.step_gan + 1) % 5 != 0:
                        #     self.step_gan += 1
                        #     continue 
                        
                        # train YOLO
                        self.model.train()
                        self.set_requires_grad(self.d3, False)
                        # pred
                        yfsource = self.model(batch['img'], gan = True)
                        # critic loss
                        # d_domain_3 = torch.sigmoid(self.d3(yfsource[2]))
                        # d_domain_3_loss = 0.5*(torch.mean((d_domain_3)**2)) # Weighted Least-squares (LS) Loss
                        d_domain_3 = self.d3(yfsource[2])
                        # d_domain_3 = torch.log_softmax(d_domain_3, 1)
                        d_domain_3_loss = self.dcgan_cri(d_domain_3, target_gt) # logsoftmax for loss

                        d_alpha = 50
                        d_domain_loss = d_alpha*(d_domain_3_loss)
                        # backward 1
                        d_domain_loss.backward() 
                        self.writer.add_scalar('domain_loss/d_domain_3_loss', d_domain_3_loss.item(), self.step_gan)
                        self.writer.add_scalar('domain_loss/d_domain_loss', d_domain_loss.item(), self.step_gan)
                        self.writer.add_scalar('pred_d/pred_d_domain_3', torch.sigmoid(d_domain_3).mean().item(), self.step_gan) 

                        # yolo loss
                        preds = self.model(batch['img'])
                        self.loss, self.loss_items = self.criterion(preds, batch)
                        self.writer.add_scalar('loss_y', self.loss.item(), self.step_gan)

                    elif self.dcgan:
                        '''
                        4: 64 160 160
                        6: 128 80 80
                        9: 256 40 40 
                        '''
                        # self.set_requires_grad(self.model, False)
                        # if epoch % 1 == 0:
                        self.set_requires_grad(self.d1, True)
                        self.set_requires_grad(self.d2, True)
                        self.set_requires_grad(self.d3, True)

                        self.model.eval()
                        with torch.no_grad():
                            yfsource = self.model(batch['img'], gan = True)
                            yftarget = self.model(batch_real['img'], gan = True)
                            
                        # visualize_feature(yfsource, "yfsource.png", batch['img'])
                        # visualize_feature(yftarget, "yftarget.png", batch_real['img'])
                        # train Critic
                        target_gt = Variable(torch.zeros(self.batch_size, 1)).to(self.device)
                        source_gt = Variable(torch.ones(self.batch_size, 1)).to(self.device)

                        
                        self.d1_optim.zero_grad()
                        self.d2_optim.zero_grad()
                        self.d3_optim.zero_grad()

                        ds_1 = self.d1(yfsource[0].detach())
                        ds_2 = self.d2(yfsource[1].detach())
                        ds_3 = self.d3(yfsource[2].detach())

                        ds_1_loss = self.dcgan_cri(ds_1, source_gt)
                        ds_2_loss = self.dcgan_cri(ds_2, source_gt)
                        ds_3_loss = self.dcgan_cri(ds_3, source_gt)

                        dt_1 = self.d1(yftarget[0].detach())
                        dt_2 = self.d2(yftarget[1].detach())
                        dt_3 = self.d3(yftarget[2].detach())

                        dt_1_loss = self.dcgan_cri(dt_1, target_gt)
                        dt_2_loss = self.dcgan_cri(dt_2, target_gt)
                        dt_3_loss = self.dcgan_cri(dt_3, target_gt)

                        c_loss = ds_1_loss + ds_2_loss + ds_3_loss + dt_1_loss + dt_2_loss + dt_3_loss
                        c_loss.backward()
                        self.d1_optim.step()
                        self.d2_optim.step()
                        self.d3_optim.step()

                        self.writer.add_scalar('loss_d/ds_3_loss', ds_3_loss.item(), self.step_gan) 
                        self.writer.add_scalar('loss_d/dt_3_loss', dt_3_loss.item(), self.step_gan) 
                        self.writer.add_scalar('loss_d/dloss', c_loss.item(), self.step_gan)
                        self.writer.add_scalar('pred_d/pred_dt_1', torch.sigmoid(dt_1).mean().item(), self.step_gan) 
                        self.writer.add_scalar('pred_d/pred_ds_1', torch.sigmoid(ds_1).mean().item(), self.step_gan)
                        self.writer.add_scalar('pred_d/pred_dt_2', torch.sigmoid(dt_2).mean().item(), self.step_gan) 
                        self.writer.add_scalar('pred_d/pred_ds_2', torch.sigmoid(ds_2).mean().item(), self.step_gan) 
                        self.writer.add_scalar('pred_d/pred_dt_3', torch.sigmoid(dt_3).mean().item(), self.step_gan) 
                        self.writer.add_scalar('pred_d/pred_ds_3', torch.sigmoid(ds_3).mean().item(), self.step_gan) 


                        # train YOLO
                        self.model.train()
                        self.set_requires_grad(self.d1, False)
                        self.set_requires_grad(self.d2, False)
                        self.set_requires_grad(self.d3, False)
                        # pred
                        yfsource = self.model(batch['img'], gan = True)
                        d_domain_1 = self.d1(yfsource[0])
                        d_domain_1_loss = self.dcgan_cri(d_domain_1, target_gt) # logsoftmax for loss
                        d_domain_2 = self.d2(yfsource[1])
                        d_domain_2_loss = self.dcgan_cri(d_domain_2, target_gt) # logsoftmax for loss
                        d_domain_3 = self.d3(yfsource[2])
                        d_domain_3_loss = self.dcgan_cri(d_domain_3, target_gt) # logsoftmax for loss


                        d_alpha = 10
                        d_domain_loss = d_alpha*(d_domain_1_loss + d_domain_2_loss + d_domain_3_loss)
                        # backward 1
                        d_domain_loss.backward()
                        
                        self.writer.add_scalar('domain_loss/d_domain_1_loss', d_domain_1_loss.item(), self.step_gan) 
                        self.writer.add_scalar('domain_loss/d_domain_2_loss', d_domain_2_loss.item(), self.step_gan) 
                        self.writer.add_scalar('domain_loss/d_domain_3_loss', d_domain_3_loss.item(), self.step_gan)
                        self.writer.add_scalar('domain_loss/d_domain_loss', d_domain_loss.item(), self.step_gan)

                        # yolo loss
                        preds = self.model(batch['img'])
                        self.loss, self.loss_items = self.criterion(preds, batch)
                        self.writer.add_scalar('loss_y', self.loss.item(), self.step_gan)

                    else:
                        preds = self.model(batch['img'])
                        self.loss, self.loss_items = self.criterion(preds, batch)

                    if rank != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items
                    
                   
                   
                self.step_gan += 1
                # Backward
                # if rgl:
                #     self.scaler.scale(self.loss + domain_t_loss + domain_s_loss).backward()
                #     # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                #     if ni - last_opt_step >= self.accumulate:
                #         self.domain_optim.step()
                #         self.optimizer_step()
                #         last_opt_step = ni
                # else:
                #     self.scaler.scale(self.loss).backward()
                #     # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                #     if ni - last_opt_step >= self.accumulate:
                #         self.optimizer_step()
                #         last_opt_step = ni
                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if rank in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if rank in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    _, testfitness = self._test_real()
                    self.writer.add_scalar('fitness test',  testfitness, epoch)  

                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                    

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if rank in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def save_model(self):
        if self.dcgan:
            torch.save(self.d1.state_dict(), self.save_dir / 'd1.pth')
            torch.save(self.d2.state_dict(), self.save_dir / 'd2.pth')
            torch.save(self.d3.state_dict(), self.save_dir / 'd3.pth')
        if self._onelayerdcgan:
            torch.save(self.d3.state_dict(), self.save_dir / 'd3.pth')
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        del ckpt

    def get_dataset(self, data):
        """
        Get train, val path from data dict if it exists. Returns None if data format is not recognized.
        """
        return data['train'], data.get('val'), data.get("test")

    def setup_model(self):
        """
        load/create/download model for any task.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type.
        """
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def _test_real(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        metrics = self.test_real(self)
        fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness


    def get_model(self, cfg=None, weights=None, verbose=True):
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        raise NotImplementedError('get_validator function not implemented in trainer')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """
        Returns dataloader derived from torch.data.Dataloader.
        """
        raise NotImplementedError('get_dataloader function not implemented in trainer')

    def criterion(self, preds, batch):
        """
        Returns loss and individual loss items as Tensor.
        """
        raise NotImplementedError('criterion function not implemented in trainer')

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        return {'loss': loss_items} if loss_items is not None else ['loss']

    def set_model_attributes(self):
        """
        To set or update model parameters before training.
        """
        self.model.names = self.data['names']

    def build_targets(self, preds, targets):
        pass

    def progress_string(self):
        return ''

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        pass

    def plot_training_labels(self):
        pass

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        pass

    def final_eval(self):
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def check_resume(self):
        resume = self.args.resume
        if resume:
            try:
                last = Path(
                    check_file(resume) if isinstance(resume, (str,
                                                              Path)) and Path(resume).exists() else get_latest_run())
                self.args = get_cfg(attempt_load_weights(last).args)
                self.args.model, resume = str(last), True  # reinstate
            except Exception as e:
                raise FileNotFoundError('Resume checkpoint not found. Please pass a valid checkpoint to resume from, '
                                        "i.e. 'yolo train resume model=path/to/last.pt'") from e
        self.resume = resume

    def resume_training(self, ckpt):
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without --resume, i.e. 'yolo task=... mode=train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            LOGGER.info('Closing dataloader mosaic')
            if hasattr(self.train_loader.dataset, 'mosaic'):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, 'close_mosaic'):
                self.train_loader.dataset.close_mosaic(hyp=self.args)

    @staticmethod
    def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        """
        Builds an optimizer with the specified parameters and parameter groups.

        Args:
            model (nn.Module): model to optimize
            name (str): name of the optimizer to use
            lr (float): learning rate
            momentum (float): momentum
            decay (float): weight decay

        Returns:
            optimizer (torch.optim.Optimizer): the built optimizer
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                    f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
        return optimizer
    @staticmethod
    def gradient_penalty(critic, fake_data, real_data, LAMBDA =10):
        device = fake_data.device
        alpha = torch.rand(fake_data.size(0), 1).to(device)
        differences = real_data - fake_data
        interpolates = fake_data + (alpha * differences)
        interpolates = torch.stack([interpolates, fake_data, real_data]).requires_grad_()

        preds = critic.last(interpolates)
        gradients = grad(preds, interpolates,
                        grad_outputs=torch.ones_like(preds),
                        retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()
        return LAMBDA*gradient_penalty

    @staticmethod
    def set_requires_grad(model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def dcgan_loss(self, pred, isreal):
        

        gt = torch.ones((self.batch_size, 1, pred.size(0), pred.size(1)), requires_grad=False) if isreal else torch.zeros((self.batch_size, 1, pred.size(0), pred.size(1)), requires_grad=False)
        
        loss = self.dcgan_cri(pred, gt)
        return loss

def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model.
    If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
    results, so AMP will be disabled during training.

    Args:
        model (nn.Module): A YOLOv8 model instance.

    Returns:
        bool: Returns True if the AMP functionality works correctly with YOLOv8 model, else False.

    Raises:
        AssertionError: If the AMP checks fail, indicating anomalies with the AMP functionality on the system.
    """
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices

    def amp_allclose(m, im):
        # All close FP32 vs AMP results
        a = m(im, device=device, verbose=False)[0].boxes.boxes  # FP32 inference
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.boxes  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    f = ROOT / 'assets/bus.jpg'  # image to check
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if ONLINE else np.ones((640, 640, 3))
    prefix = colorstr('AMP: ')
    LOGGER.info(f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...')
    try:
        from ultralytics import YOLO
        assert amp_allclose(YOLO('yolov8n.pt'), im)
        LOGGER.info(f'{prefix}checks passed ✅')
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. Setting 'amp=True'.")
    except AssertionError:
        LOGGER.warning(f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to '
                       f'NaN losses or zero-mAP results, so AMP will be disabled during training.')
        return False
    return True

    
def integrating(tensor, kernel):
    chan = tensor.size(1)
    batch = tensor.size(0)
    feature_stacks = []
    for b in range(batch):
        feature_stacks.append(torch.dstack([tensor[b,c,i::kernel,j::kernel] for i in range(kernel) for j in range(kernel) for c in range(chan)]))

    feature_stacks = torch.stack(feature_stacks)
    return feature_stacks.permute((0, 3, 1, 2))