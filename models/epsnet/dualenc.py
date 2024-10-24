import torch
import torch_geometric
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm
import math

from utils.chem import BOND_TYPES
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, generate_symmetric_edge_noise, extend_graph_order_radius, extend_to_cross_attention, extend_to_self_attention
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder, GINEncoder2, EGNN, MLPEdgeEncoder
from ..encoder.gat import GATEncoder
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform

# from diffusion import get_timestep_embedding, get_beta_schedule
import pdb

def merge_graphs_in_batch(batch1, batch2):
  merge_batch = Batch.from_data_list([val for pair in zip(batch1.to_data_list(), batch2.to_data_list()) for val in pair])
  merge_batch.graph_idx = torch.tensor(merge_batch.batch)
  merge_batch.batch = merge_batch.batch//2
  return merge_batch

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.squeeze(1)

class DualEncoderEpsNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder = MLPEdgeEncoder(128, "relu")
        self.edge_encoder2 = MLPEdgeEncoder(128, "relu")
        self.node_encoder = nn.Sequential(
                            nn.Embedding(100, 64),
                            nn.SiLU(),
                            nn.Linear(64,64),
                            )
        self.time_encoder = nn.Sequential(
                            SinusoidalPositionEmbeddings(32),
                            nn.Linear(32,32),
                            nn.SiLU(),
                            nn.Linear(32,32),
                            )
        self.template_encoder = nn.Sequential(
                                nn.Embedding(2, 32),
                                nn.SiLU(),
                                nn.Linear(32,32),
                                )
        
        self.encoder = EGNN(
            in_node_nf=128, in_edge_nf=128, hidden_nf=128, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=12, attention=True,
            tanh=False, norm_constant=0
            )

        self.encoder2 = EGNN(
            in_node_nf=128, in_edge_nf=128, hidden_nf=128, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, attention=True,
            tanh=False, norm_constant=0
            )
        
        
        # self.encoder_global = torch_geometric.nn.aggr.MLPAggregation(128, 3, 200, num_layers=3, hidden_channels=32)

        # self.encoder_attention = GATEncoder(
        #     config.hidden_dim,
        #     num_convs=6,
        # )
        
        self.encoder_cross = EGNN(
            in_node_nf=128, in_edge_nf=1, hidden_nf=128, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=8, attention=True,
            tanh=False, norm_constant=0
            )
        
        # self.encoder_global = EGNN(
        #     in_node_nf=128, in_edge_nf=1, hidden_nf=128, device='cpu',
        #     act_fn=torch.nn.SiLU(), n_layers=6, attention=False,
        #     tanh=False, norm_constant=0
        #     )
        
        # self.linear1 = nn.Linear(128, 128)
        # self.linear2 = nn.Linear(128, 128)
        
        # self.softmax  = nn.Softmax(dim=1)

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """

        '''
        Incorporate parameters together
        '''

        # denoising diffusion
        ## betas
        self.betas = nn.Parameter(torch.linspace(0.0001*1000/1000, 0.01*1000/1000, 1000), requires_grad=False)
        alphas = (1. - self.betas)#.cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)


    def forward(self, atom_type, pos, bond_index, bond_type, batch, graph_idx, time_step, template_mask,
                edge_index=None, edge_type=None, edge_length=None, return_edges=False, 
                extend_order=True, extend_radius=True, is_sidechain=None, condition=True):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=N,
            pos=pos,
            edge_index=bond_index,
            edge_type=bond_type,
            batch=graph_idx,
            order=3,
            cutoff=10,
            extend_order=True,
            extend_radius=False,
        )

        edge_index_2, edge_type_2 = extend_graph_order_radius(
            num_nodes=N,
            pos=pos,
            edge_index=bond_index,
            edge_type=bond_type,
            batch=graph_idx,
            order=3,
            cutoff=10,
            extend_order=True,
            extend_radius=True,
        )
        # edge_index = bond_index
        # edge_type = bond_type
        edge_length = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)

        if self.training:
            if (torch.rand(1)<0.9).sum()==1:
                edge_index_a = extend_to_cross_attention(pos, 200, batch, graph_idx)
            else:
                edge_index_a = extend_to_cross_attention(pos, 0, batch, graph_idx)
        else:
            if condition:
                edge_index_a = extend_to_cross_attention(pos, 200, batch, graph_idx)
            else:
                edge_index_a = extend_to_cross_attention(pos, 0, batch, graph_idx)

        # edge_length_a = get_distance(pos, edge_index_a).unsqueeze(-1)
        # edge_index_a = extend_to_cross_attention(pos, 200, batch, graph_idx)
        # local_edge_mask = is_local_edge(edge_type)  # (E, )

        

        # Encoding local
        # time_step = time_step/1000
        h = torch.cat([self.node_encoder(atom_type), self.time_encoder(time_step.index_select(0, batch).view(-1,1)), self.template_encoder(template_mask*1)], dim=1)
        # print(h.shape)
        # print(time_step.index_select(0, batch).view(-1,1).shape)
        # print(torch.cat([h, time_step.index_select(0, batch).view(-1,1)], dim=1).shape)
        # h = torch.cat([h, time_step.index_select(0, batch).view(-1,1)], dim=1)

        x = pos

        e = self.edge_encoder(
            edge_length=edge_length,
            edge_type=edge_type
        )
        # print(e.shape)

        # Local
        h, x = self.encoder(
            h = h,
            x = x,
            edges = edge_index,#[:,local_edge_mask],
            edge_attr=e,#[local_edge_mask],
            template_mask=template_mask
        ) 
        # row, col = edge_index_a
        # attention = torch.einsum("abc, dec->adc", [self.linear1(h).view(h.shape[0], -1, 8), self.linear2(h).view(h.shape[0], -1, 8)])
        # attention_mask = -torch.ones_like(attention)*1e8
        # attention_mask[row, col] = attention[row, col]
        # attention = self.softmax(attention_mask)
        
        # x[~template_mask] = pos[~template_mask]

        # print((h@h.T)[].shape)

        # print(torch.cat([attention[row, col].view(-1,8), edge_length_a], dim=1).shape)

        # h = self.encoder_attention(
        #     z=h,
        #     edge_index=edge_index_a
        # )

        edge_length_a = get_distance(x, edge_index_a).unsqueeze(-1)

        # edge_feature_a = torch.cat([attention[row, col].view(-1,8), edge_length_a], dim=1)

        # edge_feature_a = attention[row, col].view(-1,8)

        # _, x_global = self.encoder_global(
        #     h = h,
        #     x = x,
        #     edges = edge_index_a,
        #     edge_attr = edge_length_a,
        #     template_mask=template_mask
        # )
        
        # x = x + torch.index_select(torch_geometric.nn.aggr.MeanAggregation()(x_global, graph_idx), 0, graph_idx) - torch.index_select(torch_geometric.nn.aggr.MeanAggregation()(pos, graph_idx), 0, graph_idx)

        h, x = self.encoder_cross(
            h = h,
            x = x,
            edges = edge_index_a,
            edge_attr = edge_length_a,
            template_mask = template_mask
        )

        # x[~template_mask] = pos[~template_mask]

        edge_length_2 = get_distance(x, edge_index_2).unsqueeze(-1)
        e_2 = self.edge_encoder2(
            edge_length=edge_length_2,
            edge_type=edge_type_2
        )

        h, x = self.encoder2(
            h = h,
            x = x,
            edges = edge_index_2,
            edge_attr = e_2,
            template_mask=template_mask
        )

        # x_global = torch.index_select(self.encoder_global(h, graph_idx), 0, graph_idx)

        return x - pos # - torch.index_select(torch_geometric.nn.aggr.MeanAggregation()(pos, graph_idx), 0, graph_idx)
    

    def get_loss(self, ligand_batch, template_batch, 
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        N = ligand_batch.atom_type.size(0)
        num_graphs = ligand_batch.num_graphs

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs//2+1, ), device=ligand_batch.pos.device)
        time_step = torch.cat(
            [time_step, self.num_timesteps-time_step-1], dim=0)[:num_graphs]
        t = time_step
        a = self.alphas.cumprod(dim=0).index_select(0, t)
        a_pos = a.index_select(0, ligand_batch.batch).unsqueeze(-1)  # (N, 1)

        pos_noise = torch.randn_like(ligand_batch.pos)
        # pos = torch.cat([ligand_batch.pos, template_batch.pos])
        # Fix pos!
        # pos = torch.cat([val for pair in zip(ligand_batch.pos, template_batch.pos) for val in pair])
        # test용 코드
        pos = []
        for i in range(len(ligand_batch.smiles)):
            pos.append(ligand_batch[i].pos)
            pos.append(template_batch[i].pos)
        pos = torch.cat(pos)
        # print(pos.shape)
        ligand_batch.pos = ligand_batch.pos * a_pos.sqrt() + pos_noise * (1.0 - a_pos).sqrt()
        template_batch.atom_type = template_batch.atom_type + 35

        merged_batch = merge_graphs_in_batch(ligand_batch, template_batch)
        # print(template_batch.pos.max(), "!!!")
        #print(pos[:16]+pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()-ligand_batch.pos)
        template_mask = ((merged_batch.graph_idx%2)==0)
        # Fix here!

        # Update invariant edge features, as shown in equation 5-7
        x = self(
            atom_type = merged_batch.atom_type,
            pos = merged_batch.pos,
            bond_index = merged_batch.edge_index,
            bond_type = merged_batch.edge_type,
            batch = merged_batch.batch,
            graph_idx = merged_batch.graph_idx,
            time_step = time_step,
            template_mask=template_mask,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius,
            is_sidechain = is_sidechain
        )   # (E_global, 1), (E_local, 1)
        # node_eq_local = node_eq_local[template_mask]
        x = x[template_mask]

        # loss for atomic eps regression
        # loss =  ((node_eq_local + node_eq_global - pos[template_mask] - pos_noise)**2).sum()
        # eps_pos = clip_norm(eps_pos, limit=10)
        loss =  ((x - pos_noise).square()).mean()
        # loss_pos = scatter_add(loss_pos.squeeze(), node2graph)

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss
        else:
            return loss


    def langevin_dynamics_sample(self, ligand_batch, template_batch,  extend_order, extend_radius=True, 
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None, condition=True,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        
        template_batch.atom_type = template_batch.atom_type + 35
        merged_batch = merge_graphs_in_batch(ligand_batch, template_batch)
        template_mask = ((merged_batch.graph_idx%2)==0)

        pos_traj = []
        with torch.no_grad():
            # skip = self.num_timesteps // n_steps
            # seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            for i, j in tqdm(zip(reversed(seq[1:]), reversed(seq_next[1:])), desc='sample'):
                t = torch.full(size=(ligand_batch.num_graphs,), fill_value=i, dtype=torch.long, device=ligand_batch.pos.device)
                next_t = torch.full(size=(ligand_batch.num_graphs,), fill_value=j, dtype=torch.long, device=ligand_batch.pos.device)
                merged_batch = merge_graphs_in_batch(ligand_batch, template_batch)

                x = self(
                    atom_type = merged_batch.atom_type,
                    pos = merged_batch.pos,
                    bond_index = merged_batch.edge_index,
                    bond_type = merged_batch.edge_type,
                    batch = merged_batch.batch,
                    graph_idx = merged_batch.graph_idx,
                    time_step = t,
                    template_mask=template_mask,
                    return_edges = True,
                    extend_order = extend_order,
                    extend_radius = extend_radius,
                    is_sidechain = is_sidechain,
                    condition=True
                )

                # Sum
                eps_pos = x
                if clip is not None:
                    eps_pos = clip_norm(eps_pos, limit=clip)

                # Update
                
                t = t[0]
                next_t = next_t[0]

                at = self.alphas.cumprod(dim=0).index_select(0, t)
                at_next = self.alphas.cumprod(dim=0).index_select(0, next_t)

                e = eps_pos

                x0_pred = (merged_batch.pos - e*(1-at).sqrt()) / at.sqrt()
                sigma_t = ((1-at_next)/(1-at)*(1-(at/at_next))).sqrt()
                pos_next = at_next.sqrt()*x0_pred + (1-at_next-sigma_t.square()).sqrt()*e + sigma_t*torch.randn_like(merged_batch.pos)

                ligand_batch.pos = pos_next[template_mask]

                if torch.isnan(ligand_batch.pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                # ligand_batch.pos = center_pos(ligand_batch.pos, ligand_batch.batch)
                if clip_pos is not None:
                    ligand_batch.pos = torch.clamp(ligand_batch.pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(ligand_batch.pos.clone().cpu())
            
        return ligand_batch.pos, pos_traj
    
    def langevin_dynamics_sample_g(self, ligand_batch, template_batch,  extend_order, extend_radius=True, 
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        
        template_batch.atom_type = template_batch.atom_type + 35
        merged_batch = merge_graphs_in_batch(ligand_batch, template_batch)
        template_mask = ((merged_batch.graph_idx%2)==0)

        pos_traj = []
        with torch.no_grad():
            # skip = self.num_timesteps // n_steps
            # seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            for i, j in zip(reversed(seq[1:]), reversed(seq_next[1:])):
                t = torch.full(size=(ligand_batch.num_graphs,), fill_value=i, dtype=torch.long, device=ligand_batch.pos.device)
                next_t = torch.full(size=(ligand_batch.num_graphs,), fill_value=j, dtype=torch.long, device=ligand_batch.pos.device)
                merged_batch = merge_graphs_in_batch(ligand_batch, template_batch)

                x_g = self(
                    atom_type = merged_batch.atom_type,
                    pos = merged_batch.pos,
                    bond_index = merged_batch.edge_index,
                    bond_type = merged_batch.edge_type,
                    batch = merged_batch.batch,
                    graph_idx = merged_batch.graph_idx,
                    time_step = t,
                    template_mask=template_mask,
                    return_edges = True,
                    extend_order = extend_order,
                    extend_radius = extend_radius,
                    is_sidechain = is_sidechain,
                    condition=True
                )

                x_f = self(
                    atom_type = merged_batch.atom_type,
                    pos = merged_batch.pos,
                    bond_index = merged_batch.edge_index,
                    bond_type = merged_batch.edge_type,
                    batch = merged_batch.batch,
                    graph_idx = merged_batch.graph_idx,
                    time_step = t,
                    template_mask=template_mask,
                    return_edges = True,
                    extend_order = extend_order,
                    extend_radius = extend_radius,
                    is_sidechain = is_sidechain,
                    condition=False
                )

                # Sum
                eps_pos = 10*x_g - 9*x_f
                if clip is not None:
                    eps_pos = clip_norm(eps_pos, limit=clip)

                # Update
                
                t = t[0]
                next_t = next_t[0]

                at = self.alphas.cumprod(dim=0).index_select(0, t)
                at_next = self.alphas.cumprod(dim=0).index_select(0, next_t)

                e = eps_pos

                x0_pred = (merged_batch.pos - e*(1-at).sqrt()) / at.sqrt()
                sigma_t = ((1-at_next)/(1-at)*(1-(at/at_next))).sqrt()
                pos_next = at_next.sqrt()*x0_pred + (1-at_next-sigma_t.square()).sqrt()*e + sigma_t*torch.randn_like(merged_batch.pos)

                ligand_batch.pos = pos_next[template_mask]

                if torch.isnan(ligand_batch.pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                # ligand_batch.pos = center_pos(ligand_batch.pos, ligand_batch.batch)
                if clip_pos is not None:
                    ligand_batch.pos = torch.clamp(ligand_batch.pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(ligand_batch.pos.clone().cpu())
            
        return ligand_batch.pos, pos_traj

    def langevin_dynamics_sample_ddim_g(self, ligand_batch, template_batch,  extend_order, extend_radius=True, 
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        
        template_batch.atom_type = template_batch.atom_type + 35
        merged_batch = merge_graphs_in_batch(ligand_batch, template_batch)
        template_mask = ((merged_batch.graph_idx%2)==0)

        pos_traj = []
        with torch.no_grad():
            # skip = self.num_timesteps // n_steps
            # seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps

            t_max = self.num_timesteps - 1
            seq = torch.linspace(0, 1, n_steps) * t_max
            seq_prev = torch.cat([torch.tensor([-1]), seq[:-1]], dim=0)
            timesteps = reversed(seq[1:])
            timesteps_prev = reversed(seq_prev[1:])
            # return timesteps, timesteps_prev
            # seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            # seq_next = [-1] + list(seq[:-1])

            for i, j in tqdm(zip(timesteps, timesteps_prev), desc='sample'):
                t = torch.full(size=(ligand_batch.num_graphs,), fill_value=i, dtype=torch.long, device=ligand_batch.pos.device)
                next_t = torch.full(size=(ligand_batch.num_graphs,), fill_value=j, dtype=torch.long, device=ligand_batch.pos.device)
                merged_batch = merge_graphs_in_batch(ligand_batch, template_batch)

                x_g = self(
                    atom_type = merged_batch.atom_type,
                    pos = merged_batch.pos,
                    bond_index = merged_batch.edge_index,
                    bond_type = merged_batch.edge_type,
                    batch = merged_batch.batch,
                    graph_idx = merged_batch.graph_idx,
                    time_step = t,
                    template_mask=template_mask,
                    return_edges = True,
                    extend_order = extend_order,
                    extend_radius = extend_radius,
                    is_sidechain = is_sidechain,
                    condition=True
                )

                x_f = self(
                    atom_type = merged_batch.atom_type,
                    pos = merged_batch.pos,
                    bond_index = merged_batch.edge_index,
                    bond_type = merged_batch.edge_type,
                    batch = merged_batch.batch,
                    graph_idx = merged_batch.graph_idx,
                    time_step = t,
                    template_mask=template_mask,
                    return_edges = True,
                    extend_order = extend_order,
                    extend_radius = extend_radius,
                    is_sidechain = is_sidechain,
                    condition=False
                )

                # Sum
                eps_pos = 10*x_g - 9*x_f
                if clip is not None:
                    eps_pos = clip_norm(eps_pos, limit=clip)

                # Update
                
                t = t[0]
                next_t = next_t[0]

                at = self.alphas.cumprod(dim=0).index_select(0, t)
                at_next = self.alphas.cumprod(dim=0).index_select(0, next_t)

                e = eps_pos

                x0_pred = (merged_batch.pos - e*(1-at).sqrt()) / at.sqrt()
                pos_next = at_next.sqrt()*x0_pred + (1-at_next).sqrt()*e

                ligand_batch.pos = pos_next[template_mask]

                if torch.isnan(ligand_batch.pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                # ligand_batch.pos = center_pos(ligand_batch.pos, ligand_batch.batch)
                if clip_pos is not None:
                    ligand_batch.pos = torch.clamp(ligand_batch.pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(ligand_batch.pos.clone().cpu())
            
        return ligand_batch.pos, pos_traj



def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
