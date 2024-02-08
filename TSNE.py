import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.autograd import Variable
from atari_vae import TEncoder
import os, random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.init as init
from mpl_toolkits.mplot3d import Axes3D
from torchvision.models import resnet50, ResNet50_Weights


class UnFlatten(nn.Module):
  def forward(self, input, size=512):
    return input.view(input.size(0), size, 1, 1)


class ResNet(nn.Module):
  def __init__(self, embed_size=512):
    super().__init__()
    self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = self.resnet.fc.in_features
    self.resnet.fc = nn.Linear(num_ftrs, embed_size)


  def forward(self, image):
    out = self.resnet(image)
    return out


class VAEBEV(nn.Module):
  def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
    super(VAEBEV, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Flatten()
    )

    self.fc1 = nn.Linear(h_dim, z)
    self.fc2 = nn.Linear(h_dim, z)
    self.fc3 = nn.Linear(z, h_dim)

    self.decoder = nn.Sequential(
      UnFlatten(),
      nn.ConvTranspose2d(h_dim, ch * 8, kernel_size=5, stride=2),
      nn.ReLU(),
      nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=5, stride=2),
      nn.ReLU(),
      nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=6, stride=2),
      nn.ReLU(),
      nn.ConvTranspose2d(ch * 2, channel_in, kernel_size=6, stride=2),
      nn.Sigmoid(),
    )

  def reparameterize(self, mu, logvar):
    std = logvar.mul(0.5).exp_().cuda()
    # return torch.normal(mu, std)
    esp = torch.randn(*mu.size()).cuda()
    z = mu + std * esp
    return z

  def bottleneck(self, h):
    mu, logvar = self.fc1(h), self.fc2(h)
    z = self.reparameterize(mu, logvar)
    return z, mu, logvar

  def representation(self, x):
    return self.bottleneck(self.encoder(x))[0]

  def recon(self, z):
    z = self.fc3(z)
    return self.decoder(z)

  def forward(self, x):
    h = self.encoder(x)
    z, mu, logvar = self.bottleneck(h)
    return self.recon(z), mu, logvar



fpvencoder = ResNet(512).to('cuda')
#bevencoder = VAEBEV(channel_in=1, ch=16, z=512).to('cuda')
#checkpoint = torch.load('/lab/kiran/prtr4nav/FPV_BEV_TCN_CARLA_CARLA_FPVBEV_SMALL_TRAIN_STANDARD_0.1_128_512_0.0001_20.pt', map_location="cpu")
checkpoint = torch.load('/lab/kiran/prtr4nav/FPV_TCN_CARLA_CARLA_FPVBEV_SMALL_TRAIN_STANDARD_0.1_128_512_0.0001_20.pt', map_location="cpu")


#fpvencoder = ResNet(32).to('cuda')
#bevencoder = VAEBEV(channel_in=1, ch=16, z=32).to('cuda')
#checkpoint = torch.load('/lab/kiran/prtr4nav/FPV_BEV_CARLA_CARLA_FPVBEV_SMALL_TRAIN_STANDARD_0.1_128_512_0.0001_19.pt', map_location="cpu")
print(checkpoint['epoch'])

fpvencoder.load_state_dict(checkpoint['model_state_dict'])
#fpvencoder.load_state_dict(checkpoint['fpv_state_dict'])
fpvencoder.eval()
for param in fpvencoder.parameters():
    param.requires_grad = False

#bevencoder.load_state_dict(checkpoint['bev_state_dict'])
#bevencoder.eval()

#for param in bevencoder.parameters():
#    param.requires_grad = False


base_path = '/home6/tmp/kiran/'
# base_path = '/lab/tmpig10f/kiran/expert_3chan_beogym/skill2/'
games=['/carla_fpvbev_small_val/Town01_0/']



for game in games:

    rdm=[10, 20, 30 ,40 ,50]
    new=[]
    for i in range(1,5):
        new+=[q+i for q in rdm]
    rdm=new
    game_path = os.path.join(base_path, 'carla_fpvbev_small_train/Town03_0'+'/5/0/')
    bevs = np.load(game_path+'bev.npy', mmap_mode='r')
    fpvs = np.load(game_path+'fpv.npy', mmap_mode='r')
    # auxs = np.load(game_path+'aux.npy')
    ter = np.load(game_path+'terminal.npy')
    # val = np.load(game_path+'value.npy')
    ter_indices = np.where(ter == 1)[0]
    # rdm=[100, 500]
    all_bev_embedding=[]
    all_fpv_embedding=[]
    values=[]
    count=0
    for i in rdm:
        bev = torch.tensor(np.asarray(bevs[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda')
        if bev.shape[0]<3:
           continue
        count+=1
        if count==2:
            break
        # bev = bev[:,:,:,0].unsqueeze(1)
        fpv = torch.tensor(np.asarray(fpvs[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda')
        fpv = fpv.permute(0, 3, 1, 2)
        # values.append(np.asarray(val[ter_indices[i]+1 : ter_indices[i+1]]))
        values.append(np.arange(ter_indices[i+1]- ter_indices[i]-1)/(ter_indices[i+1]- ter_indices[i]-1))
        print(bev.shape)
        # bev_embeddings = bevencoder(bev.to(torch.float32)).squeeze().cpu().detach().numpy()
        fpv_embeddings = fpvencoder(fpv.to(torch.float32)).squeeze().cpu().detach().numpy()
        # all_bev_embedding.append(bev_embeddings)
        all_fpv_embedding.append(fpv_embeddings)
    # all_bev_embedding=np.concatenate(all_bev_embedding, axis=0)
    all_fpv_embedding=np.concatenate(all_fpv_embedding, axis=0)
    values=np.concatenate(values, axis=0)
    # bev_tsne = TSNE(n_components=2)
    fpv_tsne = TSNE(n_components=2)

    # new_bev_embeddings = bev_tsne.fit_transform(all_bev_embedding)
    new_fpv_embeddings = fpv_tsne.fit_transform(all_fpv_embedding)
    # plt.scatter(new_bev_embeddings[:, 0], new_bev_embeddings[:, 1], s=10, c=values, marker='o', label='Town03_0_bev')
    plt.scatter(new_fpv_embeddings[:, 0], new_fpv_embeddings[:, 1], s=10, c=values, marker='o', label='Town03_0_fpv')
    
    cbar = plt.colorbar()
    cbar.set_label('Step')
    plt.legend()
    plt.title(f'FPV BEV')
    plt.savefig(f'./Train.png')
    plt.clf()


    game_path = os.path.join(base_path, game+'5/0/')
    print(base_path)
    print(game_path)
    game_path='/home6/tmp/kiran/carla_fpvbev_small_val/Town01_1/5/0/'
    bevs = np.load(game_path+'bev.npy', mmap_mode='r')
    fpvs = np.load(game_path+'fpv.npy', mmap_mode='r')
    # auxs = np.load(game_path+'aux.npy')
    ter = np.load(game_path+'terminal.npy')
    # val = np.load(game_path+'value.npy')
    ter_indices = np.where(ter == 1)[0]
    # rdm = random.sample(range(0,len(ter_indices)-1), 50)
    # rdm=[5000,20000]
    # rdm=[20000]
    # rdm=[100, 500]
    all_bev_embedding=[]
    all_fpv_embedding=[]
    values=[]
    count=0
    for i in rdm:
        bev = torch.tensor(np.asarray(bevs[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda')
        if bev.shape[0]<3:
           continue
        count+=1
        if count==2:
            break
        fpv = torch.tensor(np.asarray(fpvs[ter_indices[i]+1 : ter_indices[i+1]])).to('cuda')
        bev = bev[:,:,:,0].unsqueeze(1)
        fpv = fpv.permute(0, 3, 1, 2)
        # values.append(np.asarray(val[ter_indices[i]+1 : ter_indices[i+1]]))
        values.append(np.arange(ter_indices[i+1]- ter_indices[i]-1)/(ter_indices[i+1]- ter_indices[i]-1))
        # bev_embeddings = bevencoder(bev.to(torch.float32)).squeeze().cpu().detach().numpy()
        fpv_embeddings = fpvencoder(fpv.to(torch.float32)).squeeze().cpu().detach().numpy()
        # all_bev_embedding.append(bev_embeddings)
        all_fpv_embedding.append(fpv_embeddings)
    # all_bev_embedding=np.concatenate(all_bev_embedding, axis=0)
    all_fpv_embedding=np.concatenate(all_fpv_embedding, axis=0)
    values=np.concatenate(values, axis=0)
    bev_tsne = TSNE(n_components=2)
    fpv_tsne = TSNE(n_components=2)

    # new_bev_embeddings = bev_tsne.fit_transform(all_bev_embedding)
    new_fpv_embeddings = fpv_tsne.fit_transform(all_fpv_embedding)
    # plt.scatter(new_bev_embeddings[:, 0], new_bev_embeddings[:, 1],  c=values,  marker='x', s= 15, linewidths=1, label=game)
    plt.scatter(new_fpv_embeddings[:, 0], new_fpv_embeddings[:, 1],  marker='o',  c=values, label=game)


    # ax.scatter(new_embeddings[:, 0], new_embeddings[:, 1], new_embeddings[:, 2], c=values, marker='x', s= 100, linewidths=3, label='SpaceInvaders')
    # plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], c=values, marker='o',s=15, label=game.split("_")[-1])

    # plt.plot(new_embeddings[:, 0], new_embeddings[:, 1], linestyle='-')
    cbar = plt.colorbar()
    cbar.set_label('Step')
    plt.legend()
    plt.title(f'FPV BEV')
    plt.savefig(f'./Val.png')
    plt.clf()
