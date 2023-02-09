# %% 

""" Created on February 8, 2023 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import scipy
import random
import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader

import os 
import time 
import json 
import pickle
import pygmt
import warnings

from scipy.sparse import (SparseEfficiencyWarning)
warnings.simplefilter('ignore', category=(FutureWarning,SparseEfficiencyWarning))
from scipy.special import softmax

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import scale, normalize, StandardScaler

import TAS as tas
import Thermobar as pt
import stoichiometry as mm

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.path as mpath
import matplotlib.colors as mcolors

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 14})
plt.rcParams['pdf.fonttype'] = 42

pt.__version__

# %% 

class FeatureDataset(Dataset):
    def __init__(self, x):
        if len(x.shape)==2:
            self.x = x
        else:
            self.x = x.reshape(-1, x.shape[-1]) #dataset keeps the right shape for training

    def __len__(self):
        return self.x.shape[0] 
    
    def __getitem__(self, n): 
        return torch.Tensor(self.x[n])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Autoencoder(nn.Module):
    def __init__(self,input_dim = 10, latent_dim = 2, hidden_layer_sizes=(512, 256, 128)):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hls = hidden_layer_sizes

        def building_block(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                # nn.LayerNorm(out_channel),
                nn.LeakyReLU(0.02),
            ]

        encoder = building_block(self.input_dim, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += building_block(self.hls[i], self.hls[i + 1])
        encoder += [nn.Linear(self.hls[-1], latent_dim)]

        decoder = building_block(latent_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += building_block(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.input_dim)] # nn.Softmax()]

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

        # print(self.encode, self.decode)

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de





class Tanh_Autoencoder(nn.Module):
    def __init__(self,input_dim = 10, latent_dim = 2, hidden_layer_sizes=(64, 32)):
        super(Tanh_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hls = hidden_layer_sizes

        def building_block(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.Tanh(),
            ]

        encoder = building_block(self.input_dim, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += building_block(self.hls[i], self.hls[i + 1])
        encoder += [nn.Linear(self.hls[-1], latent_dim)]

        decoder = building_block(latent_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += building_block(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.input_dim)] # nn.Softmax()]

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

        # print(self.encode, self.decode)

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de





def train(model, optimizer, train_loader, test_loader, n_epoch, criterion):
    
    avg_total_loss = []
    avg_test_loss = []

    for epoch in range(n_epoch):
        # Training
        model.train()
        t = time.time()
        total_loss = []
        for i, data in enumerate(train_loader):
            x = data.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.detach().item())
        
        # Testing
        model.eval()
        test_loss = []
        for i, test in enumerate(test_loader):
            x = test.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)
            test_loss.append(loss.detach().item())
        
        # Logging
        avg_loss = sum(total_loss) / len(total_loss)
        avg_test = sum(test_loss) / len(test_loss)
        avg_total_loss.append(avg_loss)
        avg_test_loss.append(avg_test)
        
        training_time = time.time() - t
        
        print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_loss:.6f}, test_loss: {avg_test:.6f}, time: {training_time:.2f} s')

    return avg_total_loss, avg_test_loss

def save_model(model, optimizer, path):
    check_point = {'params': model.state_dict(),                            
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)

def load_model(model, optimizer=None, path=''):
    check_point = torch.load(path)
    model.load_state_dict(check_point['params'])
    if optimizer is not None:
        optimizer.load_state_dict(check_point['potimizer'])

def getLatent(model, dataset:np):
    #transform real data to latent space using the trained model
    latents=[]
    model.to(device)

    dataset_ = FeatureDataset(dataset)
    loader = DataLoader(dataset_,batch_size=20,shuffle=False)
    
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(loader):
            x = data.to(device)
            z = model.encoded(x)
            latents.append(z.detach().cpu().numpy())
    
    return np.concatenate(latents, axis=0)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# %%



LEPR_AllPhases = pd.read_csv('./LEPR/LEPR_AllPhases.csv')

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
# pd.plotting.scatter_matrix(LEPR_AllPhases[oxides], figsize = (15, 15), hist_kwds={'bins':20})
plt.show()

phase_props = ['Na_K_A_Amp', 'Amp_Cation_Sum', 'Ca_CaMgFe_Cpx', 'Cpx_Cation_Sum', 
        'Na_Ca_M_Plag', 'Plag_Cation_Sum', 'Mg_Fe_M_Ol', 'Ol_Cation_Sum', 
        'Mg_Fe_M_Sp', 'Sp_Cation_Sum', 'Fe_Ti_Ox', 'Ox_Cation_Sum', 
        'Ca_P_Ap', 'Ap_Cation_Sum', 'Si_Al_T_Bt', 'Bt_Cation_Sum', 
        'Si_Al_Ti_Qz', 'Qz_Cation_Sum', 'Mg_MgFeCa_Gt', 'Gt_Cation_Sum', 
        'Na_Ca_M_Kspar', 'Kspar_Cation_Sum']
# pd.plotting.scatter_matrix(LEPR_AllPhases[phase_props], figsize = (35, 35), hist_kwds={'bins':20})


LEPR_wt = LEPR_AllPhases[oxides]
LEPR_wt = LEPR_wt.to_numpy()

LEPR_wt_norm = normalize(LEPR_wt)
ss = StandardScaler()
LEPR_wt_scale = ss.fit_transform(LEPR_wt)
LEPR_wt_softmax = softmax(LEPR_wt_norm, axis = 1)

LEPR_prop = LEPR_AllPhases[phase_props]
LEPR_prop = LEPR_prop.fillna(0).to_numpy()

LEPR_prop_norm = normalize(LEPR_prop)
ss = StandardScaler()
LEPR_prop_scale = ss.fit_transform(LEPR_prop)
LEPR_prop_softmax = softmax(LEPR_prop_norm, axis = 1)
# LEPR_wt_standardscale = ss.fit_transform(LEPR_wt)



# %% 

def autoencode(name, AE_Model, hidden_layer_sizes):

    LEPR_AllPhases = pd.read_csv('./LEPR/' + name + '.csv')
    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    LEPR_wt = LEPR_AllPhases[oxides]
    LEPR_mol = mm.calculate_mol_proportions(LEPR_wt)
    LEPR_wt = LEPR_wt.to_numpy()
    LEPR_mol = LEPR_mol.to_numpy()

    #perform z-score normalisation
    array_norm_wt = normalize(LEPR_wt)
    array_norm_mol = normalize(LEPR_mol)

    array_norm = array_norm_mol 
    ss = StandardScaler()
    array_scale = ss.fit_transform(LEPR_wt)

    # LEPR_prop = LEPR_AllPhases[phase_props]
    # LEPR_prop = LEPR_prop.fillna(0).to_numpy()

    # LEPR_prop_norm = normalize(LEPR_prop)


    # array_norm = array_norm_wt
    array_norm = array_scale


    #split the dataset into train and test sets
    train_data, test_data = train_test_split(array_norm, test_size=0.1, random_state=42)

    #define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = FeatureDataset(train_data)
    test_dataset = FeatureDataset(test_data)   

    #autoencoder params:
    lr = 1e-4
    wd = 0
    batch_size = 64
    #use half the data available
    epochs = 100
    input_size = feature_dataset.__getitem__(0).size(0)

    #define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    #define model
    model = AE_Model(input_dim=input_size, hidden_layer_sizes = hidden_layer_sizes).to(device)

    #use ADAM optimizer with mean squared error loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd) 
    criterion = nn.MSELoss()

    #train model using pre-defined function
    train_loss, test_loss = train(model, optimizer, feature_loader, test_loader, epochs, criterion)
    # np.savez('./LEPR/' + name + '_tanh_loss.npz', train_loss = train_loss, test_loss = test_loss)

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    ax.plot(np.linspace(1, epochs, epochs), train_loss, '.-', label = 'Train Loss')
    ax.plot(np.linspace(1, epochs, epochs), test_loss, '.-', label = 'Test Loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(prop={'size': 10})
    plt.tight_layout()
    # plt.savefig('./LEPR/' + name + '_tanh_testtrainloss.pdf',)

    #transform entire dataset to latent space
    z = getLatent(model, array_norm)

    phase = list(set(LEPR_AllPhases.Phase))

    tab = plt.get_cmap('tab20')
    cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
    scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

    #plot latent representation
    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    for i in range(len(phase)):
        indx = LEPR_AllPhases['Phase'] == phase[i]
        ax.scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
    ax.set_xlabel("Latent Variable 1")
    ax.set_ylabel("Latent Variable 2")
    ax.set_title(name + " Latent Space Representation")
    ax.legend(prop={'size': 10})
    plt.tight_layout()
    # plt.savefig('./LEPR/' + name + '_latentspace.pdf',)

    #save main model params
    # model_path = './LEPR/' + name + "_tanh_params.pt"
    # save_model(model, optimizer, model_path)

    #save all other params
    # conc_file = name + "_tanh.npz"
    # np.savez('./LEPR/' + name + "_tanh.npz", batch_size = batch_size, epochs = epochs, input_size = input_size, 
            # conc_file = conc_file, z = z)

    return z 

# %% 

#start execute here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
same_seeds(42)

names = ["LEPR_AllPhases"]
i = 0
start_time = time.time()
print("starting " + str(names[i]))
# autoencode(names[i], Autoencoder, (512, 256, 128))
z = autoencode(names[i], Tanh_Autoencoder, (64, 32)) # (64, 32))
print(names[i] + " done! Time: " + str(time.time() - start_time) + "s")

# %% 

# tab = plt.get_cmap('tab20')
# cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
# scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

# #plot latent representation
# fig = plt.figure(figsize = (8, 8))
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# for i in range(len(phase)):
#     indx = LEPR_AllPhases['Phase'] == phase[i]
#     ax.scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
# ax.set_xlabel("Latent Variable 1")
# ax.set_ylabel("Latent Variable 2")
# ax.set_title(" Latent Space Representation")
# # ax.legend(prop={'size': 10})
# # plt.tight_layout()

# def rotate(angle):
#     ax.view_init(azim=angle)

# rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)

# %% 


rot_animation.save('rotation.gif', dpi=80)

# %%

from matplotlib import animation

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')

def init():
    plt.cla()
    for i in range(len(phase)):
        indx = LEPR_AllPhases['Phase'] == phase[i]
        ax.scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# %% 

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=40, blit=True)
from IPython.display import HTML
HTML(anim.to_jshtml())

# %%
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')

for i in range(len(phase)):
    indx = LEPR_AllPhases['Phase'] == phase[i]
    ax.scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
    
for ii in range(0,360,1):
    ax.view_init(elev=10., azim=ii)
    # savefig("movie%d.png" % ii)

# %% 

    for i in range(len(phase)):
        indx = LEPR_AllPhases['Phase'] == phase[i]
        ax.scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])