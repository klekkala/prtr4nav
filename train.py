import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as Datasets
from itertools import cycle
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.autograd import Variable
#from evaluate import eval_adapter
from collections import defaultdict
from pytorch_metric_learning import losses
import imageio as iio
# from fancylosses import InfoNCE
from IPython import embed
from torch.hub import load_state_dict_from_url
from arguments import get_args
# from info_nce import InfoNCE
import os
import random
import numpy as np

import math
from IPython.display import clear_output
from PIL import Image
from tqdm import trange, tqdm
import utils
import initial
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import cv2
args = get_args()






if 'FPV' in args.model:
    fpvencoder, bevencoder, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(
        True)
else:
    encodernet, trainloader, div_val, start_epoch, loss_log, optimizer, device, curr_dir = initial.initialize(True)


if 'FPV' in args.model:
    #loss_func = utils.clip_loss
    loss_func = losses.ContrastiveLoss()
elif 'LSTM' in args.model and 'CARLA' in args.model:
    #loss_func = nn.MSELoss()
    loss_func = utils.gmm_loss
else:
    loss_func = utils.vae_loss


# %% Start Training
for epoch in trange(start_epoch, args.nepoch, leave=False):

    # Start the lr scheduler
    utils.lr_Linear(optimizer, args.nepoch, epoch, args.lr)
    loss_iter = []

    # contrastive case: for i, (img_batch1, img_batch2, pair) in enumerate(tqdm(trainloader, leave=False)):
    # img_batch1, img_batch2 -> [B, T, H, W]

    if 'CONT' in args.model or 'VIP' in args.model or 'VEP' in args.model or 'SOM' in args.model:
        encodernet.train()
        negiterator = iter(trainloader)
    if 'BEV_VAE' in args.model:
        encodernet.train()
        anchors = []
        for i in range(11):
            img = cv2.imread("manual_label/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=(0, 1))
            img_val = torch.tensor(img).to(device) / 255.0
            anchors.append(img_val)
    elif 'FPV' in args.model:
        fpvencoder.train()
        bevencoder.train()
    else:
        encodernet.train()


    for i, traindata in enumerate(tqdm(trainloader, leave=False)):
        if 'LSTM' in args.model:
            # CHEN
            (img, target, action) = traindata
            if len(action[0]) <= 1:
                continue
            ids, sim, image_embed = encodernet.encoder(img[0])
            source = encodernet.encoder.anchors[ids]
            source = torch.tensor(source)

            image_reshape_val = source.to(device) / div_val
            targ = target.to(device) / div_val
            action = action.to(device)
            z_gt, _, _ = encodernet.encode(targ)
            z_prev, _, _ = encodernet.encode(image_reshape_val.unsqueeze(0).unsqueeze(2))

            #mask = random.sample(range(0, len(z_prev[0])), int(len(z_prev[0]) / 2))
            #for z in z_prev:
            #    z[mask] = latent_cls[random.randint(0, 9)].to(device)
            
            encodernet.init_hs(len(img))

            #z_pred = encodernet(action, z_prev)
            #loss = loss_func(z_pred, z_gt)
            mus, sigmas, logpi = encodernet(action, z_prev)
            loss = loss_func(z_gt, mus, sigmas, logpi) / z_gt.shape[-1]

        elif "FPV_RECONBEV_CARLA" in args.model:
            (img, target) = traindata
            image_val = img.to(device) / div_val
            targ = target.to(device) / div_val

            image_embed = fpvencoder(image_val)
            #_, targ_embed_mu, targ_embed_logvar = bevencoder(targ)
            #targ_embed = targ_embed_mu
            #torch.concat((targ_embed_mu, targ_embed_logvar), axis=-1)
            #embedclasses = torch.arange(start=0, end=img.shape[0])

            # in the constrastive case, we get a batch of pair of embeddings and wheather they are positive or negative
            #loss = loss_func(torch.cat([image_embed, targ_embed], axis=0), torch.cat([embedclasses, embedclasses], axis=0))
            #loss = loss_func(image_embed, targ_embed, args.temperature, embedclasses)
            
            # reconstruct from fpv embedding
            z = bevencoder.reparameterize(image_embed[:, :32], image_embed[:, 32:])
            recon_data = bevencoder.recon(z)
            #logvar = torch.zeros(image_embed.shape).to(device)

            loss = utils.vae_loss(recon_data, targ, image_embed[:, :32], image_embed[:, 32:], args.kl_weight)

            # regression
#            loss = F.mse_loss(image_embed, targ_embed)


        elif 'FPV' in args.model:
            (img, target) = traindata
            image_val = img.to(device) / div_val
            targ = target.to(device) / div_val

            image_embed = fpvencoder(image_val)
            _, targ_embed_mu, targ_embed_logvar = bevencoder(targ)
            targ_embed = targ_embed_mu
            #torch.concat((targ_embed_mu, targ_embed_logvar), axis=-1)
            embedclasses = torch.arange(start=0, end=img.shape[0])

            # in the constrastive case, we get a batch of pair of embeddings and wheather they are positive or negative
            loss = loss_func(torch.cat([image_embed, targ_embed], axis=0), torch.cat([embedclasses, embedclasses], axis=0))
            #loss = loss_func(image_embed, targ_embed, args.temperature, embedclasses)
            
            # regression
            #loss = F.mse_loss(image_embed, targ_embed)
        else:
            (img, target) = traindata
            image_val = img.to(device) / div_val
            targ = target.to(device) / div_val
            recon_data, mu, logvar = encodernet(image_val)

            # in the constrastive case, we get a batch of pair of embeddings and wheather they are positive or negative
            loss = loss_func(recon_data, targ, mu, logvar, args.kl_weight)
        

        #print(np.mean(np.array(loss_iter)))
        # from IPython import embed; embed()
        if 'FPV' in args.model:
            fpvencoder.zero_grad()
            bevencoder.zero_grad()
        else:
            encodernet.zero_grad()
        loss.backward()
        optimizer.step()



    print("learning rate:", optimizer.param_groups[0]['lr'])
    print(np.mean(np.array(loss_iter)))

    if 'VAE' in args.model:
        auxval = str(args.kl_weight)
    else:
        auxval = str(args.temperature)

    # continue
    # Save a checkpoint with a specific filename
    if True:
        print("saving checkpoint")
        if 'FPV' in args.model:
            save_dict = {
                'epoch': epoch,
                'loss_log': loss_log,
                'fpv_state_dict': fpvencoder.state_dict(),
                'bev_state_dict': bevencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
        else:
            save_dict = {
                'epoch': epoch,
                'loss_log': loss_log,
                'model_state_dict': encodernet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }    
        torch.save(save_dict, args.save_dir + args.model + "_" + (args.expname).upper() + "_" + (args.arch).upper() + "_" + auxval + "_" + str(args.train_batch_size) + "_" + str(args.sample_batch_size) + "_" + str(args.lr) + "_" + str(epoch) + ".pt")
