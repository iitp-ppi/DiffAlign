import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm import tqdm
from easydict import EasyDict
from utils.dataloader import DataLoader

from models.epsnet import *
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
#device = 'cuda:0'
device = 'cpu'
torch.set_printoptions(precision=2, sci_mode=False)

# init model
config_path = 'configs/DiffAlign.yml'
with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))
model = get_model(config.model).to(device)
# model.load_state_dict(torch.load("./param/non_self_align/199.pt", map_location=device))#, strict=False) #199

data_path = '/home/jychoi9809/project4/data/self_align/GEOM_new/qm9_processed/geodiff_ver/qm9_1.pkl'
#data_path = 'cross_set_v2/all_align.pkl'
# data_path = 'cross_set_v2/hybrid_medium.pkl'
transforms = CountNodesPerGraph()
val_set = ConformationDataset(data_path, transform=None)
keys = ["atom_type_r", "edge_index_r", "edge_type_r", "pos_r", "smiles_r", "rdmol_r"]

val_tmp = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=4)

# test training 1 data
from torch.nn.utils import clip_grad_norm_
from utils.common import get_optimizer, get_scheduler
optimizer = get_optimizer(config.train.optimizer, model)
scheduler = get_scheduler(config.train.scheduler, optimizer)

optimizer.param_groups[0]['lr'] = 1e-4

iter_num  = 10000

model.train()
for it in range(iter_num):
    losses = []
    for batch in val_tmp:
        optimizer.zero_grad()
        #batch = next(val_it).to(device)
        batch = batch.to_data_list()
        batch2 = copy.deepcopy(batch)
        for i in range(len(batch)):
            for key in keys:
                del batch2[i][key[:-2]]
                batch2[i][key[:-2]] = batch[i][key]
                del batch[i][key]
                del batch2[i][key]
        batch = Batch.from_data_list(batch)
        batch2 = Batch.from_data_list(batch2)
        # print(batch.edge_index.max(), batch.pos.shape, batch2.atom_type.shape)
        # print(batch2.edge_index.max(), batch2.pos.shape, batch2.atom_type.shape)
        loss = model.get_loss(
            ligand_batch=copy.deepcopy(batch).to(device),
            template_batch=copy.deepcopy(batch2).to(device),
            anneal_power=config.train.anneal_power,
            return_unreduced_loss=True
        )
        # print(loss)
        if torch.isnan(loss):
            print("nan detected...")
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            # orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            losses.append(loss.item())
    loss_mean = np.mean(losses)
    print(f'[iter {it+1}], loss: {loss_mean:.4f}')
    # print to tdqm bar
    scheduler.step(loss_mean)
    if it%1 == 0:
        torch.save(model.state_dict(), f'./param/non_self_align/{it+1}.pt',)
