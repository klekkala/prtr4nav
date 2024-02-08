import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as Datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from collections import defaultdict
import imageio as iio
from IPython import embed
from torch.hub import load_state_dict_from_url
import os
import random
import numpy as np
import math
#from ray.rllib.policy.policy import Policy
from IPython.display import clear_output
#from ray.rllib.models import ModelCatalog
from PIL import Image
from tqdm import trange, tqdm
from models.RES_VAE import TEncoder as ResEncoder
from models.atari_vae import VAE, VAEBEV
from models.LSTM import StateLSTM
from models.LSTM import MDLSTM as BEVLSTM
from models.resnet import ResNet
from models.BEVEncoder import BEVEncoder
from models.atari_vae import Encoder, TEncoder, TBeoEncoder
from dataclass import BaseDataset, CarlaBEV, PosContLSTM, NegContLSTM, PosContThreeLSTM, NegContThreeLSTM, CarlaFPVBEV, CarlaFPVBEVTCN, CarlaFPVTCN
import utils
from arguments import get_args

args = get_args()


def initialize(is_train):
    if 'CARLA' in args.model:
        root_dir = "/home6/tmp/kiran/"
    elif 'BEV_LSTM' in args.model:
        root_dir = "/home6/tmp/kiran/"
    else:
        #root_dir = "/dev/shm/"
        root_dir = "/home6/tmp/kiran/"
    curr_dir = os.getcwd()
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device(args.gpu_id if use_cuda else "cpu")
    print(device)
    # %%



    transform = T.Compose([T.ToTensor()])

    #Training BEV Encoder
    if args.model == "BEV_VAE_CARLA":
        trainset = CarlaBEV.CarlaBEV(root_dir=root_dir + args.expname, transform=transform)
        encodernet = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        div_val = 255.0


    #Training LSTM encoder using BEV encoder
    elif args.model == "BEV_LSTM_CARLA":
        trainset = SingleChannelLSTM.SingleChannelLSTM(root_dir=root_dir + args.expname, transform=transform)
        vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        vae.load_state_dict(vae_ckpt['model_state_dict'])
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        
        encodernet = BEVLSTM(latent_size=32, action_size=2, hidden_size=256, gaussian_size=5,
                             num_layers=1, vae=vae).to(device)
        div_val = 255.0


    #Training FPV BEV Encoder
    elif args.model == "FPV_BEV_CARLA":
        trainset = CarlaFPVBEV.CarlaFPVBEV(root_dir=root_dir + args.expname, transform=transform)
        #this will give me a tuple: (fpv_batch, bev_batch), each of the batches are of size batch_size
        #CHEN
        #resnet150
        fpvencoder = ResNet(32).to(device)
        
        #vaeencoder
        bevencoder = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        bevencoder.load_state_dict(vae_ckpt['model_state_dict'])
        bevencoder.eval()
        for param in bevencoder.parameters():
            param.requires_grad = False

        print(root_dir, args.expname)
        div_val = 255.0


    #Training FPV BEV Encoder
    elif args.model == "FPV_BEV_TCN_CARLA":

        #negset = NegContThreeChan.NegContThreeChan(root_dir=root_dir + args.expname, transform=transform)
        trainset = CarlaFPVBEVTCN.CarlaFPVBEVTCN(root_dir=root_dir + args.expname, transform=transform, pos_distance=args.max_len, truncated=False)

        fpvencoder = ResNet(512).to(device)
        
        #vaeencoder
        bevencoder = VAEBEV(channel_in=1, ch=16, z=512).to(device)
        
        #vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        #vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        #bevencoder.load_state_dict(vae_ckpt['model_state_dict'])
        
        #I UNFROZE IT!!!!!
        #bevencoder.eval()
        #for param in bevencoder.parameters():
        #    param.requires_grad = False

        print(root_dir, args.expname)
        div_val = 255.0

    elif args.model == "FPV_TCN_CARLA":

        #negset = NegContThreeChan.NegContThreeChan(root_dir=root_dir + args.expname, transform=transform)
        trainset = CarlaFPVTCN.CarlaFPVTCN(root_dir=root_dir + args.expname, transform=transform, pos_distance=args.max_len, truncated=False)

        fpvencoder = ResNet(512).to(device)
        
        print(root_dir, args.expname)
        div_val = 255.0

    else:
        raise ("Not Implemented Error")

    # %%
    # get a test image batch from the testloader to visualise the reconstruction quality
    # dataiter = iter(testloader)
    # test_images, _ = dataiter.next()



    if is_train:
        trainloader, _ = utils.get_data_STL10(trainset, args.train_batch_size, transform)
    elif is_train == False and 'LSTM' in args.model:
        trainloader, _ = utils.get_data_STL10(trainset, 1, transform)
        args.load_checkpoint = True
    else:
        trainloader, _ = utils.get_data_STL10(trainset, 20, transform)
        args.load_checkpoint = True
    
    
    # setup optimizer
    if 'FPV_BEV' in args.model:
        optimizer = optim.Adam(list(fpvencoder.parameters()) + list(bevencoder.parameters()), lr=args.lr, betas=(0.5, 0.999))
    elif 'FPV_' in args.model:
        optimizer = optim.Adam(list(fpvencoder.parameters()), lr=args.lr, betas=(0.5, 0.999))
    else:
        optimizer = optim.Adam(encodernet.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # Loss function
    loss_log = []



    # Create the results directory if it does note exist
    if not os.path.isdir(curr_dir + "/Results"):
        os.makedirs(curr_dir + "/Results")


    # Load Checkpoint
    if args.load_checkpoint:
        if 'VAE' in args.model:
            auxval = args.kl_weight
        else:
            auxval = args.temperature
        if args.model_path == "":
            model_path = "FPV_BEV_CARLA_CARLA_FPVBEV_SMALL_TRAIN_STANDARD_0.1_128_512_0.0001_12.pt"
            #this one will throw a bug!!!!
            #model_path = args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (
            #    args.arch).upper() + "_" + str(auxval) + "_" + str(args.sgamma) + "_" + str(
            #    args.train_batch_size) + "_" + str(args.sample_batch_size) + "_" + str(args.lr) + "_" + str(args.nepoch) + ".pt"
        else:
            model_path = args.save_dir + args.model_path
        
        print(model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        print("Checkpoint loaded")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        start_epoch = checkpoint["epoch"]
        print("Epoch starting at ", start_epoch)
        loss_log = checkpoint["loss_log"]
        
        if 'FPV' in args.model:
            fpvencoder.load_state_dict(checkpoint['fpv_state_dict'])
            bevencoder.load_state_dict(checkpoint['bev_state_dict'])
        else:
            encodernet.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If checkpoint does exist raise an error to prevent accidental overwriting
        # if os.path.isfile(args.save_dir + args.model + ".pt"):
        #    raise ValueError("Warning Checkpoint exists. Overwriting")
        # else:
        #    print("Starting from scratch")
        start_epoch = 0


    #Return all the collected variables
    if 'FPV_BEV' in args.model or "FPV_RECONBEV_CARLA" in args.model:
        return fpvencoder, bevencoder, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir    
    elif 'FPV_' in args.model:
        return fpvencoder, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir    
    else:
        return encodernet, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir
