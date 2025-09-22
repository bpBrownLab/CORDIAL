#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader

class DataLoaderFactory:
    """Factory for creating DataLoaders with dataset-specific collate functions."""

    collate_fn_dict = {
        'none': None,  # Use default collating function (i.e., just stacking the data)
        'generic': None,
        'interaction_graph_legacy': None
    }

    batch_types = {'full-batch', 'mini-batch', 'online-learning'}

    @classmethod
    def get_collate_fn(cls, alias):
        """
        Get collate function by alias.

        Args:
            alias: Collate function alias

        Returns:
            Collate function or None for default behavior

        Raises:
            ValueError: If alias is not recognized
        """
        if alias in cls.collate_fn_dict:
            return cls.collate_fn_dict[alias]
        else:
            raise ValueError(f"Unknown collate function alias: {alias}")

    @classmethod
    def create_data_loader(cls, dataset, alias,
                           batch_method='mini-batch', batch_size=32, sampler=None,
                           shuffle=True, pin_memory=False, num_workers=0,
                           **kwargs
                           ):
        """
        Create DataLoader with dataset-specific configuration.

        Args:
            dataset: Dataset to load
            alias: Collate function alias
            batch_method: Batching strategy
            batch_size: Batch size
            sampler: Optional sampler
            shuffle: Whether to shuffle data
            pin_memory: Whether to pin memory for GPU
            num_workers: Number of worker processes
            **kwargs: Additional arguments

        Returns:
            Configured DataLoader
        """
        if batch_method not in cls.batch_types:
            raise ValueError("Invalid batch method specified!")
        
        # Update batch size
        if batch_method == 'online-learning':
            batch_size = 1
        elif batch_method == 'full-batch':
            batch_size = len(dataset)
        elif batch_method == 'mini-batch':
            batch_size = min(batch_size, len(dataset))
        else:
            raise ValueError("Invalid batch method specified!")
        
        # Ensure batch_size is not larger than the dataset
        dataset_size = len(dataset)
        print(f"Dataset size: {dataset_size}")
        print(f"Batch size: {batch_size}")
        print(f"Dataset type: {type(dataset)}")
        
        if isinstance(dataset, torch.utils.data.Subset):
            print(f"Subset indices length: {len(dataset.indices)}")
            print(f"Original dataset length: {len(dataset.dataset)}")

        print(f"collate fn alias: {alias}")
        print(f"collate fn: {cls.get_collate_fn(alias)}")

        collate_fn = cls.get_collate_fn(alias)

        # No custom collate function; stick with default
        if alias in ['generic', 'interaction_graph_legacy']:
            # Only use shuffle if no sampler is provided
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                              shuffle=(shuffle if sampler is None else False), 
                              pin_memory=pin_memory, num_workers=num_workers, drop_last=False)

        elif alias in ['interaction_graph'] or collate_fn:
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                              shuffle=(shuffle if sampler is None else False), 
                              pin_memory=pin_memory, num_workers=num_workers, drop_last=False,
                              collate_fn=lambda batch: collate_fn(batch))
        
        # Keep this last!
        elif alias in ['none']:
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                              shuffle=(shuffle if sampler is None else False), 
                              pin_memory=pin_memory, num_workers=num_workers, drop_last=False)
        else:
            raise ValueError(f"Unknown collate function alias: {alias}")

