import math
import torch

from torch import nn
from tqdm.auto import tqdm
from torch_geometric.data import Batch

from ..encoder import EGNN, MLPEdgeEncoder
from ..common import extend_graph_order_radius, extend_to_cross_attention


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def merge_graphs_in_batch(batch1, batch2):
  merge_batch = Batch.from_data_list([val for pair in zip(batch1.to_data_list(), batch2.to_data_list()) for val in pair])
  merge_batch.graph_idx = torch.tensor(merge_batch.batch)
  merge_batch.batch = merge_batch.batch//2
  return merge_batch


class SinusoidalTimeEmbeddings(nn.Module):
    """
    Sinusoidal Time Embedder.

    Args:
        out_dim (int): The output dimension of the embedding.
    """
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, time):
        """
        Args:
            time (torch.tensor): A tensor of shaped `(num_nodes, 1)` representing time values of each node.

        returns:
            torch.tensor: A tensor of shape `(num_nodes, self.out_dim)` containing the sinusoidal time embeddings.
        """
        device = time.device
        half_out_dim = self.out_dim // 2
        embeddings = math.log(10000) / (half_out_dim - 1)
        embeddings = torch.exp(torch.arange(half_out_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.squeeze(1)


class DiffAlign(nn.Module):

    def __init__(self):
        super().__init__()
        self.edge_encoder = MLPEdgeEncoder(128, "relu")
        self.edge_encoder2 = MLPEdgeEncoder(128, "relu")
        
        self.node_encoder = nn.Sequential(
                            nn.Embedding(100, 64),
                            nn.SiLU(),
                            nn.Linear(64,64),
                            )
        
        self.time_encoder = nn.Sequential(
                            SinusoidalTimeEmbeddings(32),
                            nn.Linear(32,32),
                            nn.SiLU(),
                            nn.Linear(32,32),
                            )
        
        self.query_encoder = nn.Sequential(
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
        
        self.encoder_cross = EGNN(
            in_node_nf=128, in_edge_nf=1, hidden_nf=128, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=8, attention=True,
            tanh=False, norm_constant=0
            )

        self.betas = nn.Parameter(torch.linspace(0.0001*1000/1000, 0.01*1000/1000, 1000), requires_grad=False)
        self.alphas = nn.Parameter((1. - self.betas), requires_grad=False)
        self.num_timesteps = self.betas.size(0)


    def forward(self, query_batch, reference_batch, time_step, condition=True):
        """
        Args:
            quey_batch (torch_geometric.data.Batch): A batch for query molecules containg atom_type, edge_index, edge_type, pos, and batch as attributes.
            reference_batch (torch_geometric.data.Batch): A batch for reference molecules with the same structure as query_batch.
            time_step (torch.tensor): A time step index vector for each node shaped `(num_batches,)`.
            condition (bool, optional): A flag indicating whether to apply conditioning. Defaults to `True`.

        Returns:
            torch.tensor: Predicted noise of shape `(n_nodes, 3)`.
            
        """

        merged_batch = merge_graphs_in_batch(query_batch, reference_batch)

        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=merged_batch.atom_type.size(0),
            pos=merged_batch.pos,
            edge_index=merged_batch.edge_index,
            edge_type=merged_batch.edge_type,
            batch=merged_batch.graph_idx,
            order=3,
            cutoff=10,
            extend_order=True,
            extend_radius=False,
        )

        edge_index_2, edge_type_2 = extend_graph_order_radius(
            num_nodes=merged_batch.atom_type.size(0),
            pos=merged_batch.pos,
            edge_index=merged_batch.edge_index,
            edge_type=merged_batch.edge_type,
            batch=merged_batch.graph_idx,
            order=3,
            cutoff=10,
            extend_order=True,
            extend_radius=True,
        )
        
        edge_length = get_distance(merged_batch.pos, edge_index).unsqueeze(-1)   # (E, 1)
        query_mask = ((merged_batch.graph_idx%2)==0)

        # Create cross-attention-like edges to apply conditions.
        if self.training:
            if (torch.rand(1)<0.9).sum()==1:
                edge_index_a = extend_to_cross_attention(merged_batch.pos, 200, merged_batch.batch, merged_batch.graph_idx)
            else:
                edge_index_a = extend_to_cross_attention(merged_batch.pos, 0, merged_batch.batch, merged_batch.graph_idx)
        else:
            if condition:
                edge_index_a = extend_to_cross_attention(merged_batch.pos, 200, merged_batch.batch, merged_batch.graph_idx)
            else:
                edge_index_a = extend_to_cross_attention(merged_batch.pos, 0, merged_batch.batch, merged_batch.graph_idx)

        h = torch.cat([self.node_encoder(merged_batch.atom_type), self.time_encoder(time_step.index_select(0, merged_batch.batch).view(-1,1)), self.query_encoder(query_mask*1)], dim=1)

        x = merged_batch.pos

        e = self.edge_encoder(
            edge_length=edge_length,
            edge_type=edge_type
        )

        h, x = self.encoder(
            h = h,
            x = x,
            edges = edge_index,
            edge_attr=e,
            coord_mask=query_mask
        ) 

        edge_length_a = get_distance(x, edge_index_a).unsqueeze(-1)

        h, x = self.encoder_cross(
            h = h,
            x = x,
            edges = edge_index_a,
            edge_attr = edge_length_a,
            coord_mask = query_mask
        )

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
            coord_mask=query_mask
        )

        return x[query_mask] - merged_batch.pos[query_mask]
    

    def get_loss(self, query_batch, reference_batch):
        """
        Args:
            quey_batch (torch_geometric.data.Batch): A batch for query molecules containg atom_type, edge_index, edge_type, pos, and batch as attributes.
            reference_batch (torch_geometric.data.Batch): A batch for reference molecules with the same structure as query_batch.

        Returns:
            torch.tensor: Calculated loss, a scalar tensor.
            
        """
        time_step = torch.randint(0, self.num_timesteps, size=(query_batch.num_graphs, ), device=query_batch.pos.device)
        a = self.alphas.cumprod(dim=0).index_select(0, time_step)
        a_pos = a.index_select(0, query_batch.batch).unsqueeze(-1)  # (N, 1)

        # Add noise as ddpm manner
        pos_noise = torch.randn_like(query_batch.pos)
        query_batch.pos = query_batch.pos * a_pos.sqrt() + pos_noise * (1.0 - a_pos).sqrt()
        reference_batch.atom_type = reference_batch.atom_type + 35

        x = self(
            query_batch=query_batch,
            reference_batch=reference_batch,
            time_step = time_step,
        )

        loss =  ((x - pos_noise).square()).mean()
        return loss


    def DDPM_Sampling(self, query_batch, reference_batch):
        """
        Args:
            quey_batch (torch_geometric.data.Batch): A batch for query molecules containg atom_type, edge_index, edge_type, pos, and batch as attributes.
            reference_batch (torch_geometric.data.Batch): A batch for reference molecules with the same structure as query_batch.

        Returns:
            torch.tensor: Predicted position of query molecule shaped `(num_query_nodes, 3)`.
            list[torch.tensor]: Trajectory of query molecule, contatining 999 tensors each shaped `(num_query_nodes, 3)`.
        """
        reference_batch.atom_type = reference_batch.atom_type + 35

        pos_traj = []
        with torch.no_grad():
            seq = range(0, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            for i, j in tqdm(zip(reversed(seq[1:]), reversed(seq_next[1:])), desc='sample'):
                t = torch.full(size=(query_batch.num_graphs,), fill_value=i, dtype=torch.long, device=query_batch.pos.device)
                next_t = torch.full(size=(query_batch.num_graphs,), fill_value=j, dtype=torch.long, device=query_batch.pos.device)

                x = self(
                    query_batch=query_batch,
                    reference_batch=reference_batch,
                    time_step=t,
                    condition=True,
                )

                eps_pos = x
                
                at = self.alphas.cumprod(dim=0).index_select(0, t[0])
                at_next = self.alphas.cumprod(dim=0).index_select(0, next_t[0])

                e = eps_pos

                # Denoising as DDPM manner
                x0_pred = (query_batch.pos - e*(1-at).sqrt()) / at.sqrt()
                sigma_t = ((1-at_next)/(1-at)*(1-(at/at_next))).sqrt()
                pos_next = at_next.sqrt()*x0_pred + (1-at_next-sigma_t.square()).sqrt()*e + sigma_t*torch.randn_like(query_batch.pos)

                query_batch.pos = pos_next

                if torch.isnan(query_batch.pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                
                pos_traj.append(query_batch.pos.clone().cpu())
            
        return query_batch.pos, pos_traj
    
    def DDPM_CFG_Sampling(self, query_batch, reference_batch):
        """
        Args:
            quey_batch (torch_geometric.data.Batch): A batch for query molecules containg atom_type, edge_index, edge_type, pos, and batch as attributes.
            reference_batch (torch_geometric.data.Batch): A batch for reference molecules with the same structure as query_batch.

        Returns:
            torch.tensor: Predicted position of query molecule shaped `(num_query_nodes, 3)`.
            list[torch.tensor]: Trajectory of query molecule, contatining 999 tensors each shaped `(num_query_nodes, 3)`.
        """
        reference_batch.atom_type = reference_batch.atom_type + 35

        pos_traj = []
        with torch.no_grad():
            seq = range(0, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            for i, j in tqdm(zip(reversed(seq[1:]), reversed(seq_next[1:]))):
                t = torch.full(size=(query_batch.num_graphs,), fill_value=i, dtype=torch.long, device=query_batch.pos.device)
                next_t = torch.full(size=(query_batch.num_graphs,), fill_value=j, dtype=torch.long, device=query_batch.pos.device)

                x_g = self(
                    query_batch=query_batch,
                    reference_batch=reference_batch,
                    time_step = t,
                    condition=True
                )

                x_f = self(
                    query_batch=query_batch,
                    reference_batch=reference_batch,
                    time_step = t,
                    condition=False
                )

                eps_pos = 10*x_g - 9*x_f

                at = self.alphas.cumprod(dim=0).index_select(0, t[0])
                at_next = self.alphas.cumprod(dim=0).index_select(0, next_t[0])

                e = eps_pos

                # Denoising as DDPM manner
                x0_pred = (query_batch.pos - e*(1-at).sqrt()) / at.sqrt()
                sigma_t = ((1-at_next)/(1-at)*(1-(at/at_next))).sqrt()
                pos_next = at_next.sqrt()*x0_pred + (1-at_next-sigma_t.square()).sqrt()*e + sigma_t*torch.randn_like(query_batch.pos)

                query_batch.pos = pos_next

                if torch.isnan(query_batch.pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()

                pos_traj.append(query_batch.pos.clone().cpu())
            
        return query_batch.pos, pos_traj

    def DDIM_CFG_Sampling(self, query_batch, reference_batch, n_steps):
        """
        Args:
            quey_batch (torch_geometric.data.Batch): A batch for query molecules containg atom_type, edge_index, edge_type, pos, and batch as attributes.
            reference_batch (torch_geometric.data.Batch): A batch for reference molecules with the same structure as query_batch.
            n_steps (int): A number of steps for DDIM sampling.

        Returns:
            torch.tensor: Predicted position of query molecule shaped `(num_query_nodes, 3)`.
            list[torch.tensor]: Trajectory of query molecule, contatining 999 tensors each shaped `(num_query_nodes, 3)`.
        """
        reference_batch.atom_type = reference_batch.atom_type + 35

        pos_traj = []
        with torch.no_grad():

            t_max = self.num_timesteps - 1
            seq = torch.linspace(0, 1, n_steps) * t_max
            seq_prev = torch.cat([torch.tensor([-1]), seq[:-1]], dim=0)
            timesteps = reversed(seq[1:])
            timesteps_prev = reversed(seq_prev[1:])

            for i, j in tqdm(zip(timesteps, timesteps_prev), desc='sample'):
                t = torch.full(size=(query_batch.num_graphs,), fill_value=i, dtype=torch.long, device=query_batch.pos.device)
                next_t = torch.full(size=(query_batch.num_graphs,), fill_value=j, dtype=torch.long, device=query_batch.pos.device)

                x_g = self(
                    query_batch=query_batch,
                    reference_batch=reference_batch,
                    time_step = t,
                    condition=True
                )

                x_f = self(
                    query_batch=query_batch,
                    reference_batch=reference_batch,
                    time_step = t,
                    condition=False
                )

                eps_pos = 10*x_g - 9*x_f

                at = self.alphas.cumprod(dim=0).index_select(0, t[0])
                at_next = self.alphas.cumprod(dim=0).index_select(0, next_t[0])

                e = eps_pos

                # Denoising as DDIM manner
                x0_pred = (query_batch.pos - e*(1-at).sqrt()) / at.sqrt()
                pos_next = at_next.sqrt()*x0_pred + (1-at_next).sqrt()*e

                query_batch.pos = pos_next

                if torch.isnan(query_batch.pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos_traj.append(query_batch.pos.clone().cpu())
            
        return query_batch.pos, pos_traj