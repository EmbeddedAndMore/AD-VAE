import os
import glob
from collections import defaultdict
import random
from abc import ABC, abstractmethod
import importlib
import functools
import time


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
import cv2 as cv
import imagesize
import torchvision.transforms as T
from PIL import Image
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import utils
from dataset import TrashDataset



def load_model(model_name):
    """import model using model name"""
    path = "models." + model_name
    model = importlib.import_module(path)
    model = model.__dict__[model_name.upper()]

    if model is None:
        print(" There is no model")
        exit(0)
    return model

def create_model(opt):
    """create model with model name"""
    model = load_model(opt.model)
    model = model(opt)
    print("model [%s] was created" % type(model).__name__)
    return model

def init_weights(net, init_type='normal', init_gain=0.02):
    """ define the initialization function and batchnorms"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu=[], mode='train'):
    """Initilaizing networks
    If # gpus is more than 1, we would be better to use dataparallel for keeping from memory shortage.
    And if this model is train mode, we need to initialize weights for rasing performance"""
    if len(gpu) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu[0])
        net = nn.DataParallel(net, gpu)
    if mode == 'train':
        init_weights(net, init_type, init_gain=init_gain)
        return net

    return net

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler




class Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            # State (100x1x1)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(True),

            # State (64x64x64)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(True),

            # State (32x128x128)
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, input):
        img = self.model(input)
        return img.view(img.shape[0], *self.img_shape)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            # State (3x256x256)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (16x128x128)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (32x64x64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (64x32x32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x8x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=1, padding=0)
            # output of main module --> State (1024x1x1)
        )

        self.last_layer = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )


    def forward(self, img):
        features = self.model(img)
        features = features.view(img.shape[0], -1)
        features = self.last_layer(features)
        features = features.view(features.shape[0], -1, 1, 1)
        return features
#------------------------------------------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feature):
        feature = feature.flatten(1)
        value = self.model(feature)
        return value

class Encoder_aae(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            # State (3x256x256)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (16x128x128)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (32x64x64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (64x32x32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x8x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=1, padding=0)
            # output of main module --> State (1024x1x1)
        )

        self.last_layer = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.mu_layer = nn.Linear(512, latent_dim, bias=True)
        self.std_layer = nn.Linear(512, latent_dim, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, img):
        z = self.model(img)
        features = z.view(img.shape[0], -1)
        features = self.last_layer(features)
        mu = self.mu_layer(features)
        logvar = self.mu_layer(features)
        z = self.reparameterize(mu, logvar)
        z = z.view(features.shape[0], -1, 1, 1)
        return z




class BaseModel(ABC):

    def __init__(self, opt):
        self.opt = opt
        self.gpu = opt.gpu
        self.device = torch.device(f'cuda:{self.gpu[0]}') if self.gpu else torch.device('cpu')
        self.optimizers = []
        self.networks = []
        self.save_dir = os.path.join(opt.save_dir, opt.object)
        if self.opt.mode == 'Train':
            self.isTrain = True
        elif self.opt.mode == 'Pretrained' or self.opt.mode == 'Test':
            self.isTrain = False
    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def test(self):
        pass

    def setup(self, opt):
        if opt.mode == 'train':
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        elif opt.mode == 'test':
            self.load_networks()
        self.print_networks(opt.verbose)

    def set_requires_grad(self, *nets, requires_grad=False):
        for _, net in enumerate(nets):
            for param in net.parameters():
                param.requires_grad = requires_grad

    def get_generated_imags(self):
        visual_imgs = None
        for name in self.visual_names:
            if isinstance(name, str):
                visual_imgs = getattr(self, name)
        return visual_imgs

    def eval(self):
        for name in self.networks:
            net = getattr(self, name)
            net.eval()

    def update_learning_rate(self, epoch):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'{epoch} : learning rate {old_lr:.7f} -> {lr:.7f}')
    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.networks:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self):
        utils.mkdirs(self.save_dir)
        save_encoder_filename = f'{self.model_name}_e.pth'
        save_decoder_filename = f'{self.model_name}_d.pth'
        save_encoder_path = os.path.join(self.save_dir, save_encoder_filename)
        save_decoder_path = os.path.join(self.save_dir, save_decoder_filename)
        net_d = getattr(self, 'decoder')
        net_e = getattr(self, 'encoder')

        if len(self.gpu) > 0 and torch.cuda.is_available():
            torch.save(net_d.module.cpu().state_dict(), save_decoder_path)
            net_d.cuda(self.gpu[0])
            torch.save(net_e.module.cpu().state_dict(), save_encoder_path)
            net_e.cuda(self.gpu[0])
            print(f"saved model (gpu) in:  {save_encoder_path} - {save_decoder_path}")

        else:
            torch.save(net_d.cpu().state_dict(), save_decoder_path)
            torch.save(net_e.cpu().state_dict(), save_encoder_path)
            print(f"saved model  in:  {save_encoder_path} - {save_decoder_path}")


    def load_networks(self):
        load_encoder_filename = f'{self.model_name}_e.pth'
        load_decoder_filename = f'{self.model_name}_d.pth'
        load_encoder_path = os.path.join(self.save_dir, load_encoder_filename)
        load_decoder_path = os.path.join(self.save_dir, load_decoder_filename)
        net_e = getattr(self, 'encoder')
        net_d = getattr(self, 'decoder')
        if isinstance(net_d, torch.nn.DataParallel):
            net_d = net_d.module
        if isinstance(net_e, torch.nn.DataParallel):
            net_e = net_e.module
        print('loading the encoder from %s' % load_encoder_path)
        print('loading the decoder from %s' % load_decoder_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        encoder_state_dict = torch.load(load_encoder_path)
        decoder_state_dict = torch.load(load_decoder_path)

        net_e.load_state_dict(encoder_state_dict)
        net_d.load_state_dict(decoder_state_dict)


    def get_current_losses(self, *loss_name):
        loss = {}
        for name in loss_name:
            loss[name] = (float(getattr(self, name)))  # float(...) works for both scalar tensor and float number
        return loss


class AAE(BaseModel):

    @staticmethod
    def add_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        return parser

    def __init__(self, opt):
        """Initialize the CAE model"""
        BaseModel.__init__(self, opt)
        self.opt = opt
        img_size = (self.opt.channels, self.opt.img_size, self.opt.img_size)
        latent = self.opt.latent
        self.encoder = init_net(Encoder_aae(latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize encoder networks doing data parallel and init_weights
        self.decoder = init_net(Decoder(img_size, latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize decoder networks doing data parallel and init_weights
        self.discriminator = init_net(Discriminator(latent).cuda(), gpu=opt.gpu, mode=opt.mode)
        # initialize discriminator networks doing data parallel and init_weights
        self.networks = ['encoder', 'decoder', 'discriminator']
        self.criterion = torch.nn.MSELoss()
        self.criterion_dm = torch.nn.BCELoss()
        self.visual_names = ['generated_imgs']
        self.model_name = self.opt.model
        self.loss_name = ['recon_loss', 'dm_loss', 'g_loss']
        self.real_label = torch.ones([self.opt.batch_size, 1])
        self.fake_label = torch.zeros([self.opt.batch_size, 1])
        if self.opt.mode == 'train':# if mode is train, we have to set optimizer and requires grad is true
            self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_dm = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr/5,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_e)
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_dm)
            self.set_requires_grad(self.decoder, self.encoder, self.discriminator, requires_grad=True)

    def forward_recon(self):
        z = self.encoder(self.real_imgs)
        self.generated_imgs = self.decoder(z)

    def forward_dm(self):
        z_fake = self.encoder(self.real_imgs)
        self.fake = self.discriminator(z_fake)
        z_real_gauss = torch.randn(self.real_imgs.size()[0], self.opt.latent)
        self.real = self.discriminator(z_real_gauss)

        self.real_label = self.real_label.type_as(self.real)
        self.fake_label = self.fake_label.type_as(self.fake)

    def backward_recon(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.eval()
        self.recon_loss = self.criterion(10.*self.real_imgs, 10.*self.generated_imgs)
        self.recon_loss.backward()

    def backward_dm(self):
        # discriminator train
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()
        self.dm_loss = self.criterion_dm(self.real, self.real_label) + self.criterion_dm(self.fake, self.fake_label)
        # print("dm_loss: ", self.dm_loss)
        self.dm_loss.backward()

    def backward_g(self):
        # generator train
        self.encoder.train()
        self.discriminator.eval()
        self.fake = self.discriminator(self.encoder(self.real_imgs))
        self.g_loss = self.criterion_dm(self.fake, self.real_label)
        self.g_loss.backward()

    def set_input(self, input):
        self.real_imgs = input.to(self.device)

    def train(self):
        # recon train
        self.forward_recon()
        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()
        self.backward_recon()
        self.optimizer_d.step()
        self.optimizer_e.step()

        # discriminator train
        self.forward_dm()
        self.optimizer_dm.zero_grad()
        self.backward_dm()
        self.optimizer_dm.step()

        # generator train
        self.optimizer_d.zero_grad()
        self.backward_g()
        self.optimizer_d.step()

    def test(self):
        with torch.no_grad():
            self.forward_recon()

    def save_images(self, data):
        images = data['img']
        paths = os.path.join(self.opt.save_dir, self.opt.object)
        paths = os.path.join(paths, "result")
        anomaly_img = utils.compare_images(images, self.generated_imgs, threshold=self.opt.threshold)
        utils.save_images(anomaly_img, paths, data)


def get_transform(opt):
    """Transforms images"""
    transform = []
    if opt.rotate == True:
        transform.append(T.RandomRotation(0.5))
    transform.append(T.ColorJitter(brightness=opt.brightness))
    transform.append(T.Resize((opt.cropsize, opt.cropsize), interpolation=2))
    transform.append(T.ToTensor())
    if opt.channels == 3:
        transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif opt.channels == 1:
        transform.append(T.Normalize((0.5), (0.5)))

    return T.Compose(transform)


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, dataset):
        """Create a dataset instance given the name [dataset_mode] and a multi-threaded data loader."""
        self.opt = opt
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            num_workers=int(opt.num_threads),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # if i * self.opt.batch_size >= self.opt.max_dataset_size:
            #     break
            yield data


class Opt:
    data_dir: str = "../student_db"
    channels:int = 3
    img_size: int = 256
    gpu: list[int] = [0]
    model: str = "aae"
    save_dir: str = "vae_v2_saved_models/aae"
    latent: int = 100
    init_type: str = "normal" # [normal | xavier | kaiming | orthogonal]
    init_gain: float = 0.02
    cropsize: int = 256
    verbose:bool=True

    # train options
    print_epoch_freq: int = 1
    save_epoch_freq: int = 50
    epoch_count: int = 0
    n_epochs: int = 60
    n_epochs_decay: int = 500
    beta1: float = 0.5
    beta2: float = 0.999
    lr: float = 0.00002
    lr_policy: str = "linear" # [linear | step | plateau | cosine]
    lr_decay_iters: int = 50
    batch_size: int = 128
    mode: str = "train"
    num_threads:int = 10
    no_dropout: bool = True
    rotate: bool = False
    brightness:float=  0.1
    object:str = "notrash"


if __name__ == "__main__":
    opt = Opt()
    opt.model = "test"
    if opt.mode == "train":
        print("Running Train")
        print(f"PID: {os.getpid()}")

        dataset = TrashDataset("../student_db", transform=get_transform(opt), train=True, grayscale=False)
        dataset_size = len(dataset)
        data_loader = CustomDatasetDataLoader(opt, dataset)
        dataset = data_loader.load_data()
        print(f"Training size is = {dataset_size}")

        model = AAE(opt)       # create model (AE, AAE)
        model.setup(opt)   
        total_iters = 0
        loss_name = model.loss_name            # loss name for naming 

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            epoch_start_time = time.time()                      # start epoch time
            model.update_learning_rate(epoch)                   # update learning rate change with schedulers
            epoch_iters = 0

            for i, (data, _) in enumerate(dataset):                  # dataset loop
                iter_start_time = time.time()                   # start iter time
                model.set_input(data)                           # unpacking input data for processing
                model.train()                                   # start model train
                total_iters += 1
                epoch_iters += 1
            if epoch % opt.print_epoch_freq == 0:               # model loss, time print frequency
                losses = model.get_current_losses(*loss_name)
                epoch_time = time.time() - epoch_start_time
                message = f"epoch : {epoch} | total_iters : {total_iters} | epoch_time:{epoch_time:.3f}"
                for k,v in losses.items():
                    message += f" | {k}:{v}"
                print(message)
            if epoch % opt.save_epoch_freq == 0:                # save model frequency
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                model.save_networks()
                utils.plt_show(model.generated_imgs[:3])
        model.save_networks()
    
    else:
        print("Running Test")
        opt.threshold = 0.2    # get test options
        dataset = TrashDataset("../student_db", transform=get_transform(opt), train=True, grayscale=False)
        dataset_size = len(dataset)
        data_loader = CustomDatasetDataLoader(opt, dataset)
        dataset = data_loader.load_data()
        print(f"Training size is = {dataset_size}")
        model = AAE(opt)       # create model (AE, AAE)
        model.setup(opt)                # set model : if mode is 'train', define schedulers and if mode is 'test', load saved networks
        model.eval()                    # model eval version
        for i, (data,_) in enumerate(dataset):
            epoch_start_time = time.time()
            model.set_input(data)
            model.test()

            generated_images = model.get_generated_imags()

            epoch_time = time.time() - epoch_start_time
            print(f"{i} epoch_time : {epoch_time:.3f}")
            model.save_images(data)
        print("end Test")