import os
import hydra
import torch
import numpy as np
import pkg_resources
import multiprocessing
import psutil
from typing import Union, List
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool

from graphium.finetuning.fingerprinting import Fingerprinter
from graphium.config._loader import (
    load_accelerator,
    load_predictor,
    load_metrics,
    load_architecture,
    load_datamodule,
)

class MinimolParallel:
    
    def __init__(self, batch_size: int = None):
        self.num_cpus = multiprocessing.cpu_count()
        self.batch_size = batch_size or self.estimate_optimal_batch_size()
        if batch_size is None: print(f"Using dynamically determined batch_size of {batch_size}.")
        else: print(f"Using user-defined batch size of {batch_size}.")
        self.num_workers = max(1, self.num_cpus - 2)  # Leave some CPUs free

        # Handle the paths
        state_dict_path = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'state_dict.pth')
        config_path = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'config.yaml')
        base_shape_path = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'base_shape.yaml')

        # Load config
        cfg = self.load_config(os.path.basename(config_path))
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg['accelerator']['type'] = 'cpu'  # Force CPU processing

        self.cfg, accelerator_type = load_accelerator(cfg)
        self.cfg['architecture']['mup_base_path'] = base_shape_path

        # Load datamodule
        self.datamodule = load_datamodule(self.cfg, accelerator_type)

        # Load model
        model_class, model_kwargs = load_architecture(cfg, in_dims=self.datamodule.in_dims)
        metrics = load_metrics(self.cfg)

        predictor = load_predictor(
            config=self.cfg,
            model_class=model_class,
            model_kwargs=model_kwargs,
            metrics=metrics,
            task_levels=self.datamodule.get_task_levels(),
            accelerator_type=accelerator_type,
            featurization=self.datamodule.featurization,
            task_norms=self.datamodule.task_norms,
            replicas=1,
            gradient_acc=1,
            global_bs=self.datamodule.batch_size_training,
        )
        self.set_training_mode_false(predictor)
        predictor.load_state_dict(torch.load(state_dict_path), strict=False)
        self.predictor = Fingerprinter(predictor, 'gnn:15')
        self.predictor.setup()

    def estimate_optimal_batch_size(self) -> int:
        """Estimate the largest possible batch size based on available RAM."""
        total_mem = psutil.virtual_memory().total
        available_mem = psutil.virtual_memory().available
        max_batch_size = int((available_mem / total_mem) * 100_000)  # Adjust based on memory fraction
        return max(10_000, min(max_batch_size, 500_000))  # Ensure a reasonable range

    def set_training_mode_false(self, module):
        if isinstance(module, torch.nn.Module):
            module.training = False
            for submodule in module.children():
                self.set_training_mode_false(submodule)
        elif isinstance(module, list):
            for value in module:
                self.set_training_mode_false(value)
        elif isinstance(module, dict):
            for _, value in module.items():
                self.set_training_mode_false(value)

    def load_config(self, config_name):
        hydra.initialize('ckpts/minimol_v1/', version_base=None)
        cfg = hydra.compose(config_name=config_name)
        return cfg

    def featurize_batch(self, smiles_batch: List[str]):
        """Fully parallel featurization with error handling using threads inside each process."""
        
        def featurize_single(smi):
            try:
                input_features, _ = self.datamodule._featurize_molecules([smi])
                input_features = self.to_fp32(input_features)
                return input_features[0] if input_features else None
            except Exception:
                return None  # Skip failed featurization

        valid_features = []
        
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):  # Suppress output
            with ThreadPoolExecutor(max_workers=min(8, len(smiles_batch))) as executor:  # Limit thread count
                results = list(executor.map(featurize_single, smiles_batch))

        # Remove failed entries
        valid_features = [res for res in results if res is not None]
        
        return valid_features

    def __call__(self, smiles: Union[str, List[str]]) -> List[torch.Tensor]:
        smiles = [smiles] if isinstance(smiles, str) else smiles

        results = []
        num_batches = len(smiles) // self.batch_size + (len(smiles) % self.batch_size > 0)

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            batch_iter = (smiles[i: i + self.batch_size] for i in range(0, len(smiles), self.batch_size))
            for valid_features in tqdm(executor.map(self.featurize_batch, batch_iter), total=num_batches, desc="Featurizing SMILES"):
                if valid_features:  # Ignore empty batches
                    batch = Batch.from_data_list(valid_features)
                    batch = {"features": batch, "batch_indices": batch.batch}
                    node_features = self.predictor.get_fingerprints_for_batch(batch)
                    fingerprint_graph = global_max_pool(node_features, batch['batch_indices'])
                    num_molecules = fingerprint_graph.shape[0]
                    results.extend(fingerprint_graph[:num_molecules])

        return results

    def to_fp32(self, input_features: list) -> list:
        """Convert tensors to float32 while handling errors."""
        failures = 0
        for input_feature in input_features:
            try:
                if isinstance(input_feature, dict):
                    for k, v in input_feature.items():
                        if isinstance(v, torch.Tensor):
                            if v.dtype == torch.half:
                                input_feature[k] = v.float()
                            elif v.dtype == torch.int32:
                                input_feature[k] = v.long()
                else:
                    failures += 1
            except Exception:
                failures += 1
                continue  # Skip on failure

        return [f for f in input_features if f]  # Remove failed features
