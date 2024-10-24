#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm import tqdm
from easydict import EasyDict
from rdkit.Chem import AllChem

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

os.chdir('..')
os.getcwd()

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.chem import *
#device = 'cuda:0'
device = 'cpu'
torch.set_printoptions(precision=2, sci_mode=False)


# In[2]:


os.environ["CUDA_LAUNCH_BLOCKING"]="1"


# In[3]:


# init model
config_path = '/home/jychoi9809/projects_octo/DiffAlign/configs/DiffAlign.yml'
with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))
model = get_model(config.model).to(device)


# In[4]:


data_path = '/home/jychoi9809/project4/data/self_align/GEOM_new/qm9_processed/geodiff_ver/qm9_1.pkl'
val_set = ConformationDataset(data_path, transform=None)


# In[5]:


#from utils.dataloader import DataLoader
from torch_geometric.data import DataLoader

val_tmp = DataLoader(val_set, batch_size=1, shuffle=False)
#val_it = iter(val_tmp)
val_it = inf_iterator(val_tmp)
for i in range(5):
    batch = next(val_it)
batch = batch.to(device)


# In[6]:


import py3Dmol


# In[7]:


def show(mol):
  mblock = Chem.MolToMolBlock(mol)
  view = py3Dmol.view(width=500, height=500)
  view.addModel(mblock, 'mol')
  view.setStyle({'stick':{}})
  view.zoomTo()
  view.show()


# In[8]:


model.load_state_dict(torch.load("/home/jychoi9809/projects_octo/DiffAlign/param/2000.pt"))


# In[9]:


# inference
n_steps = 1000 

batch1 = batch.to_data_list()
batch2 = copy.deepcopy(batch1)
mean_pos = batch2[0]['pos'].mean(0)
batch2[0]['pos'] = batch2[0]['pos'] - mean_pos
ligand_batch = Batch.from_data_list(batch1)
template_batch = Batch.from_data_list(batch2)
model.eval()

ligand_batch.pos.normal_()

pos_gen, pos_gen_traj = model.langevin_dynamics_sample_g(
    ligand_batch=ligand_batch,
    extend_order=True, # Done in transforms.
    n_steps=n_steps,
    step_lr=1e-6,
    w_global=0.9,
    global_start_sigma=1.0,
    clip=1000.0,
    clip_local=None,
    sampling_type='generalized',
    #sampling_type='generalized',
    eta=1.0
)
pos_gen = pos_gen.cpu() + mean_pos.cpu()


# In[10]:


index = 0
#mol = rdkit.Chem.MolFromSmiles(batch[index]['smiles'])
mol = batch[index]['rdmol']
#mol = rdkit.Chem.rdmolops.AddHs(mol)
#AllChem.EmbedMultipleConfs(mol)
init_mol = set_rdmol_positions(mol, batch[index]['pos'])

mblock = Chem.MolToMolBlock(init_mol)
with open(f'/home/jychoi9809/projects_octo/DiffAlign/init.sdf', 'w') as f:
  f.write(mblock)

view = py3Dmol.view(width=500, height=500)
view.addModel(mblock, 'mol')
view.setStyle({'stick':{}})
view.zoomTo()
view.show()

traj_len = len(pos_gen_traj)

for i in np.linspace(0, traj_len-1, 100).astype(int):
    pos = pos_gen_traj[i].cpu()
    out_mol = set_rdmol_positions(init_mol, pos[(batch.batch == 0)])
    mblock1 = Chem.MolToMolBlock(out_mol)
    with open(f'/home/jychoi9809/projects_octo/DiffAlign/step/{i}.sdf', 'w') as f:
      f.write(mblock1)

#mol2 = rdkit.Chem.MolFromSmiles(ligand_batch[index]['smiles'])
mol2 = ligand_batch[index]['rdmol']
#mol2 = rdkit.Chem.rdmolops.AddHs(mol2)
#AllChem.EmbedMultipleConfs(mol2)
gen_mol = set_rdmol_positions(mol2, pos_gen)#[ligand_batch.ptr[index]:ligand_batch.ptr[index+1]])

mblock = Chem.MolToMolBlock(gen_mol)

with open(f'/home/jychoi9809/projects_octo/DiffAlign/result.sdf', 'w') as f:
  f.write(mblock)

view = py3Dmol.view(width=500, height=500)
view.addModel(mblock, 'mol')
view.setStyle({'stick':{}})
view.zoomTo()
view.show()


# In[13]:


gen_mol = set_rdmol_positions(gen_mol, pos_gen)
mblock = Chem.MolToMolBlock(init_mol)
view = py3Dmol.view(width=500, height=500)
view.addModel(mblock, 'mol')
mblock = Chem.MolToMolBlock(gen_mol)
view.addModel(mblock, 'mol')
view.setStyle({'model':0},{'stick':{'color': 'orange'}})
view.setStyle({'model':1},{'stick':{'color': 'green'}})
view.zoomTo()
view.show()


# In[ ]:





# In[15]:


# print traj
traj_len = len(pos_gen_traj)
for i in np.linspace(0, traj_len-1, 10).astype(int):
  pos = pos_gen_traj[i].cpu()
  out_mol = set_rdmol_positions(init_mol, pos)
  print(f'step {i}')
  show(out_mol)


# In[ ]:




