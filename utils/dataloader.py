from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
import copy
from torch.utils.data.dataloader import default_collate

from torch_geometric.data.collate import collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter

class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.keys = ["atom_type_r", "edge_index_r", "edge_type_r", "pos_r", "smiles_r", "rdmol_r"]

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            batch2 = copy.deepcopy(batch)
            for i in range(len(batch)):
                for key in self.keys:
                    del batch2[i][key[:-2]], batch2[i][key]
                    batch2[i][key[:-2]] = batch[i][key]
                    del batch[i][key]
            # batch = Batch.from_data_list(batch, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
            batch2 = Batch.from_data_list(batch2, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
            for i in range(len(batch)):
                for key in self.keys:
                    # print(batch2[i][key[:-2]])
                    # print(key)
                    batch[i][key] = batch2[i][key[:-2]]
            # for key in self.keys + ["batch_r", "ptr_r"]:
            #     batch[key] = batch2[key[:-2]]
            return Batch.from_data_list(batch, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )
