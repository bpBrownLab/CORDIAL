#!/usr/bin/env python

"""
GPU-accelerated version of interaction_graph_property_distance_featurizer.py

This provides massive speedup by vectorizing all distance binning, property computations,
and histogram accumulation operations on GPU.
"""

import torch
import time
from modules.features.compute_properties import unified_compute_atomic_properties

class InteractionGraphPropertyDistanceFeaturizerGPU:
    """GPU-accelerated featurizer for molecular interaction graphs."""
    
    def __init__(self, ligand, protein, interaction_graph_builder,
                 step_size=0.25, num_distance_bins=64, property_pairs=None,
                 device='cuda'):
        """
        Initialize GPU-accelerated molecular interaction featurizer.

        Args:
            ligand, protein: Input molecules
            interaction_graph_builder: Pre-built interaction graph
            step_size: Distance bin width (angstroms)
            num_distance_bins: Number of distance bins  
            property_pairs: Property pair specifications
            device: Compute device
        """
        init_start_time = time.time()
        self.device = torch.device(device)
        self.step_size = step_size
        self.num_distance_bins = num_distance_bins
        self.property_pairs = property_pairs
        
        # Get interaction data
        self.interaction_graph = interaction_graph_builder.interaction_graph
        self.distances = interaction_graph_builder.atom_pair_distances
        self.interacting_atoms1 = interaction_graph_builder.interacting_atoms1
        self.interacting_atoms2 = interaction_graph_builder.interacting_atoms2
        
        # Move to GPU
        if hasattr(self.interaction_graph, 'to'):
            self.interaction_graph = self.interaction_graph.to(self.device)
        else:
            self.interaction_graph = torch.tensor(self.interaction_graph, device=self.device)
            
        if hasattr(self.distances, 'to'):
            self.distances = self.distances.to(self.device)
        else:
            self.distances = torch.tensor(self.distances, device=self.device)
        
        # Compute properties (still need CPU for RDKit)
        self.mol1_properties, self.mol2_properties = self._compute_properties(ligand, protein)
        
        # GPU-accelerated feature computation
        self.feature_tensor = self._gpu_compute_features()
        
        self.num_feature_columns = self.feature_tensor.size(-1)
        print(f"[TIMER] InteractionGraphPropertyDistanceFeaturizerGPU.__init__ total time: {time.time() - init_start_time:.4f} seconds")

    def _compute_properties(self, ligand, protein):
        """Compute atomic properties on CPU (RDKit limitation)"""
        compute_props_start = time.time()
        keys = list(self.property_pairs.keys())
        
        mol1_properties = unified_compute_atomic_properties(
            ligand, [k[0] for k in keys], [protein], 
            self.step_size * self.num_distance_bins
        )
        mol2_properties = unified_compute_atomic_properties(
            protein, [k[1] for k in keys], [ligand], 
            self.step_size * self.num_distance_bins
        )
        
        # Convert to GPU tensors
        conversion_start = time.time()
        mol1_props_gpu = {}
        mol2_props_gpu = {}
        
        for prop_name, values in mol1_properties.items():
            # Vectorized gathering of properties for interacting atoms
            all_values_t = torch.tensor(values, device=self.device)
            interacting_indices_t = torch.tensor(self.interacting_atoms1, dtype=torch.long, device=self.device)
            mol1_props_gpu[prop_name] = all_values_t[interacting_indices_t]
            
        for prop_name, values in mol2_properties.items():
            # Vectorized gathering of properties for interacting atoms
            all_values_t = torch.tensor(values, device=self.device)
            interacting_indices_t = torch.tensor(self.interacting_atoms2, dtype=torch.long, device=self.device)
            mol2_props_gpu[prop_name] = all_values_t[interacting_indices_t]
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  [TIMER] Property conversion and GPU transfer time: {time.time() - conversion_start:.4f}s")
        print(f"  [TIMER] _compute_properties total time: {time.time() - compute_props_start:.4f}s")
        return mol1_props_gpu, mol2_props_gpu

    def _gpu_compute_features(self):
        """
        GPU-accelerated feature computation using vectorized operations
        """
        gpu_compute_start = time.time()
        # Get all interaction indices at once
        interaction_mask = self.interaction_graph == 1
        interaction_indices = torch.nonzero(interaction_mask, as_tuple=False)
        
        if len(interaction_indices) == 0:
            # No interactions - return zeros
            total_feature_cols = self._get_total_feature_columns()
            return torch.zeros(self.num_distance_bins, total_feature_cols, device=self.device)
        
        # Vectorized distance binning
        interaction_distances = self.distances[interaction_indices[:, 0], interaction_indices[:, 1]]
        distance_bins = torch.round(interaction_distances / self.step_size).long()
        
        # Filter valid distance bins
        valid_mask = (distance_bins >= 0) & (distance_bins < self.num_distance_bins)
        interaction_indices = interaction_indices[valid_mask]
        distance_bins = distance_bins[valid_mask]
        
        if len(interaction_indices) == 0:
            total_feature_cols = self._get_total_feature_columns()
            return torch.zeros(self.num_distance_bins, total_feature_cols, device=self.device)
        
        # Initialize feature tensor
        total_feature_cols = self._get_total_feature_columns()
        features = torch.zeros(self.num_distance_bins, total_feature_cols, device=self.device)
        
        # Process each property pair
        col_offset = 0
        for (prop1_name, prop2_name), binning_scheme in self.property_pairs.items():
            features, col_offset = self._process_property_pair_gpu(
                features, col_offset, interaction_indices, distance_bins,
                prop1_name, prop2_name, binning_scheme
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  [TIMER] _gpu_compute_features total time: {time.time() - gpu_compute_start:.4f}s")
        return features

    def _process_property_pair_gpu(self, features, col_offset, interaction_indices, 
                                 distance_bins, prop1_name, prop2_name, binning_scheme):
        """
        GPU-accelerated processing of a single property pair
        """
        # Get property values for all interactions
        mol1_vals = self.mol1_properties[prop1_name][interaction_indices[:, 0]]
        mol2_vals = self.mol2_properties[prop2_name][interaction_indices[:, 1]]
        
        # Compute products
        products = mol1_vals * mol2_vals
        
        # Filter non-zero products
        nonzero_mask = products != 0.0
        if not nonzero_mask.any():
            return features, col_offset + self._get_binning_factor(binning_scheme)
        
        # Apply non-zero mask
        filtered_distance_bins = distance_bins[nonzero_mask]
        filtered_products = products[nonzero_mask]
        filtered_mol1_vals = mol1_vals[nonzero_mask]
        filtered_mol2_vals = mol2_vals[nonzero_mask]
        
        if binning_scheme == 'unsigned':
            # Simple accumulation - use scatter_add for proper shape handling
            values = filtered_products.float()
            features[:, col_offset].scatter_add_(0, filtered_distance_bins, values)
            col_offset += 1
            
        elif binning_scheme == 'signed':
            # Three bins: -/-, +/+, -/+
            products_abs = torch.abs(filtered_products).float()
            
            # Vectorized sign logic
            neg_neg_mask = (filtered_mol1_vals < 0) & (filtered_mol2_vals < 0)
            pos_pos_mask = (filtered_mol1_vals >= 0) & (filtered_mol2_vals >= 0)
            opp_sign_mask = (filtered_mol1_vals * filtered_mol2_vals < 0)
            
            # Accumulate each bin type using scatter_add
            if neg_neg_mask.any():
                features[:, col_offset].scatter_add_(0, filtered_distance_bins[neg_neg_mask], 
                                                   products_abs[neg_neg_mask])
            
            if pos_pos_mask.any():
                features[:, col_offset + 1].scatter_add_(0, filtered_distance_bins[pos_pos_mask], 
                                                       products_abs[pos_pos_mask])
            
            if opp_sign_mask.any():
                features[:, col_offset + 2].scatter_add_(0, filtered_distance_bins[opp_sign_mask], 
                                                       products_abs[opp_sign_mask])
            
            col_offset += 3
            
        elif binning_scheme == 'signed_directional':
            # Four bins: -/-, +/+, -/+, +/-
            products_abs = torch.abs(filtered_products).float()
            
            # Vectorized directional sign logic
            neg_neg_mask = (filtered_mol1_vals < 0) & (filtered_mol2_vals < 0)
            pos_pos_mask = (filtered_mol1_vals >= 0) & (filtered_mol2_vals >= 0)
            neg_pos_mask = (filtered_mol1_vals < 0) & (filtered_mol2_vals >= 0)
            pos_neg_mask = (filtered_mol1_vals >= 0) & (filtered_mol2_vals < 0)
            
            # Accumulate each bin type using scatter_add
            if neg_neg_mask.any():
                features[:, col_offset].scatter_add_(0, filtered_distance_bins[neg_neg_mask], 
                                                   products_abs[neg_neg_mask])
            
            if pos_pos_mask.any():
                features[:, col_offset + 1].scatter_add_(0, filtered_distance_bins[pos_pos_mask], 
                                                       products_abs[pos_pos_mask])
            
            if neg_pos_mask.any():
                features[:, col_offset + 2].scatter_add_(0, filtered_distance_bins[neg_pos_mask], 
                                                       products_abs[neg_pos_mask])
            
            if pos_neg_mask.any():
                features[:, col_offset + 3].scatter_add_(0, filtered_distance_bins[pos_neg_mask], 
                                                       products_abs[pos_neg_mask])
            
            col_offset += 4
        
        return features, col_offset

    def _get_binning_factor(self, binning_scheme):
        """Get number of feature columns for a binning scheme"""
        factors = {'unsigned': 1, 'signed': 3, 'signed_directional': 4}
        return factors[binning_scheme]

    def _get_total_feature_columns(self):
        """Calculate total number of feature columns"""
        total = 0
        for binning_scheme in self.property_pairs.values():
            total += self._get_binning_factor(binning_scheme)
        return total

    def get_feature_tensor(self):
        """Get the computed feature tensor"""
        return self.feature_tensor

