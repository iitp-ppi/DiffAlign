import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm import tqdm
from easydict import EasyDict
from diffalign.utils.dataloader import DataLoader

from diffalign.models.epsnet import *
from diffalign.utils.datasets import ConformationDataset
from torch.nn.utils import clip_grad_norm_
from diffalign.utils.common import get_optimizer, get_scheduler
from diffalign.utils.transforms import *
from diffalign.utils.misc import *
if __name__ == "__main__":
    device = 'cuda:0'
    torch.set_printoptions(precision=2, sci_mode=False)

    # init model
    model = get_model(config.model).to(device)

    # set data_path
    data_path = 'cross_set_v2/zinc_all.pkl'
    transforms = CountNodesPerGraph()
    val_set = ConformationDataset(data_path, transform=None)
    keys = ["atom_type_r", "edge_index_r", "edge_type_r", "pos_r", "smiles_r", "rdmol_r"]

    val_tmp = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

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
            loss = model.get_loss(
                query_batch=copy.deepcopy(batch).to(device),
                reference_batch=copy.deepcopy(batch2).to(device),
            )
            if torch.isnan(loss):
                print("nan detected...")
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                losses.append(loss.item())
        loss_mean = np.mean(losses)
        print(f'[iter {it+1}], loss: {loss_mean:.4f}')

        scheduler.step(loss_mean)
        if it%1 == 0:
            torch.save(model.state_dict(), f'./param/{it+1}.pt',)
