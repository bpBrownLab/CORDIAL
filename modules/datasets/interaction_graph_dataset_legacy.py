#!/usr/bin/env python

# PyTorch imports
import torch
from torch.utils.data import Dataset

# General imports
import numpy as np
import pandas as pd
from collections import OrderedDict
import multiprocessing
import os
import pickle
import gc
import io
import time
from typing import Optional

# Project imports
from modules.representations.interaction_graph_builder import InteractionGraphBuilder
from modules.representations.interaction_graph_property_distance_featurizer_gpu import InteractionGraphPropertyDistanceFeaturizerGPU
from utils.generic_molecule_loader import load_molecule
from utils import normalization_utils

def _load_molecule_worker(path):
    """Helper function to load a single molecule, for multiprocessing."""
    try:
        molecule = load_molecule(path)
        return path, molecule
    except Exception as e:
        # It's helpful to know which file failed, even in parallel.
        print(f"Worker failed to load molecule {path}: {e}")
        return path, None

class InteractionGraphDatasetLegacy(Dataset):
    """
    A dataset class for creating interaction graphs from pairs of molecules and their interaction results.
    Supports GPU acceleration for faster processing.

    Attributes:
        search_method (str): Method used to calculate distances in the interaction graph.
        distance_cutoff (float): Maximum distance for considering interactions.
        step_size (float): Step size for distance binning.
        num_distance_bins (int): Number of bins to use for distance discretization.
        molecule_paths_1 (list): List of file paths for the first set of molecules.
        molecule_paths_2 (list): List of file paths for the second set of molecules.
        result_paths (list): List of file paths for result labels for the interactions.
        skipped_indices (list): List of indices that were skipped due to errors.
        valid_indices (list): List of valid indices in the dataset.
        device (torch.device): Device to use for computations ('cuda' or 'cpu').
        use_gpu (bool): Whether GPU acceleration is enabled and available.
    """

    def __init__(self, ligand_file=None, protein_file=None, ligand_protein_pair_file=None,
                 property_pairs=None, transform=None,
                 search_method='cdist', distance_cutoff=16.0, step_size=0.25, num_distance_bins=64,
                 reduce_interaction_graph=False, inference=True,
                 load_normalization_data_pkl=None,
                 device=None, use_gpu=True, batch_processing=True,
                 precompute_features=True, precompute_batch_size=8192,
                 cache_dir: Optional[str] = None):
        """
        Initializes the dataset with molecule pairs for inference.
        Includes feature pre-computation for GPU acceleration.

        Parameters:
            ligand_file (str, optional): Path to CSV file containing paths for the first set of molecules.
            protein_file (str, optional): Path to CSV file for the second set of molecules.
            ligand_protein_pair_file (str, optional): Path to CSV file containing paths for both sets of molecules.
            property_pairs (dict, optional): Dict of atomic property pairs to include in features. Defaults to a predefined dict.
            transform (callable, optional): Optional transform to be applied on a sample.
            search_method (str): Method used to calculate distances in the interaction graph. Defaults to 'cdist'.
            distance_cutoff (float): Maximum distance for considering interactions. Defaults to 16.0.
            step_size (float): Step size for distance binning. Defaults to 0.25.
            num_distance_bins (int): Number of bins to use for distance discretization. Defaults to 64.
            reduce_interaction_graph (bool): Whether to reduce the interaction graph. Defaults to False.
            inference (bool): Always True for this inference-only dataset.
            load_normalization_data_pkl (str, optional): Path to normalization data for feature scaling.
            device (str): Device to use ('cuda', 'cpu', or 'auto'). 'auto' will use GPU if available. Defaults to 'auto'.
            use_gpu (bool): Whether to enable GPU acceleration. Defaults to True.
            batch_processing (bool): Whether to enable batch processing. Defaults to True.
            precompute_features (bool): Whether to pre-compute all features during initialization. Defaults to True.
            precompute_batch_size (int): Batch size for GPU pre-computation. Defaults to 8192.
            cache_dir (str, optional): Directory to store pre-computed features on disk. If None, uses in-memory cache.
        """

        super(InteractionGraphDatasetLegacy, self).__init__()

        # GPU/Device setup (needs to be early for loading)
        self.use_gpu = use_gpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        else:
            self.device = device

        # Feature cache setup
        self.features_cache = {}
        loaded_from_cache = False
        
        # Cache logic: cache_dir enables on-disk caching
        self.cache_dir = cache_dir
        self.metadata_path = os.path.join(self.cache_dir, 'metadata.pkl') if self.cache_dir else None
        
        # Single-file cache implementation
        self.cache_file = os.path.join(self.cache_dir, 'features.cache') if self.cache_dir else None
        self.cache_index_path = os.path.join(self.cache_dir, 'features.index.pkl') if self.cache_dir else None
        self.cache_index = {}
        self.cache_file_handle = None

        # Cache handling logic
        loaded_from_cache = False
        self.precompute_features = precompute_features

        if self.cache_dir:
            cache_exists = os.path.exists(self.metadata_path) and os.path.exists(self.cache_index_path)

            # If cache exists and no source files are given, load from cache
            if cache_exists and ligand_protein_pair_file is None and ligand_file is None:
                print(f"Loading from on-disk cache: {self.cache_dir}")
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                with open(self.cache_index_path, 'rb') as f:
                    self.cache_index = pickle.load(f)
                
                self.valid_indices = metadata['valid_indices']
                self.precompute_batch_size = metadata.get('precompute_batch_size', precompute_batch_size)
                
                print(f"Loaded metadata for {len(self.valid_indices)} samples from cache.")
                loaded_from_cache = True
                self.precompute_features = False

        # Initialize parameters
        self.search_method = search_method
        self.distance_cutoff = distance_cutoff
        self.property_pairs = property_pairs if property_pairs is not None else self._default_property_pairs()
        self.step_size = step_size
        self.num_distance_bins = num_distance_bins
        self.transform = transform
        self.reduce_interaction_graph = reduce_interaction_graph
        
        init_start_time = time.time()
        
        if loaded_from_cache:
            self.precompute_features = False
        
        self.precompute_batch_size = precompute_batch_size

        # For normalization purposes
        self.inference = inference
        self.load_normalization_data_pkl = load_normalization_data_pkl

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Using on-disk cache for pre-computed features at: {self.cache_dir}")

        # Check if GPU is actually available and enabled
        self.gpu_available = torch.cuda.is_available() and self.device.type == 'cuda'
        if self.gpu_available:
            print(f"GPU acceleration enabled on device: {self.device}")
        else:
            print("Using CPU for computations")

        # Molecule caching is now handled per-chunk during precomputation
        self.molecule_cache = {}

        # If not loaded from cache, proceed with original file loading
        if not loaded_from_cache:
            # Read molecule file paths
            self.molecule_paths_1, self.molecule_paths_2 = self._read_molecule_paths(ligand_file, protein_file, ligand_protein_pair_file)

            # Validate molecules and create valid_indices list
            print("Validating molecules for robust loading...")
            self.valid_indices = []
            self.skipped_indices = []
            
            validation_start_time = time.time()
            for i in range(len(self.molecule_paths_1)):
                try:
                    # Test load both molecules
                    mol1 = self._get_molecule(self.molecule_paths_1[i])
                    mol2 = self._get_molecule(self.molecule_paths_2[i])
                    
                    if mol1 is not None and mol2 is not None:
                        self.valid_indices.append(i)
                    else:
                        self.skipped_indices.append(i)
                        if mol1 is None:
                            print(f"  Skipping sample {i}: Failed to load molecule 1: {self.molecule_paths_1[i]}")
                        if mol2 is None:
                            print(f"  Skipping sample {i}: Failed to load molecule 2: {self.molecule_paths_2[i]}")
                        
                except Exception as e:
                    self.skipped_indices.append(i)
                    print(f"  Skipping sample {i}: Error validating molecules: {str(e)}")
                    
            validation_time = time.time() - validation_start_time
            print(f"Molecule validation completed in {validation_time:.2f}s")
            print(f"Valid samples: {len(self.valid_indices)}/{len(self.molecule_paths_1)}")
            print(f"Skipped samples: {len(self.skipped_indices)}")
            
            if len(self.skipped_indices) > 0:
                print(f"WARNING: {len(self.skipped_indices)} samples were skipped due to molecule loading issues.")
                print("This is normal for datasets with problematic molecules. Inference will continue with valid samples.")
        
        # Cached dataset initialization
        if loaded_from_cache:
            self.skipped_indices = []
            print("Dataset initialized from self-contained cache.")
        
        self.batch_processing = batch_processing and self.gpu_available
        
        # Feature pre-computation
        if self.precompute_features:
            if self.cache_dir is None:
                print("WARNING: Pre-computing features without a cache_dir. All features will be stored in RAM.")

            if self.gpu_available:
                print("Pre-computing features using GPU acceleration...")
                self._precompute_features_gpu()
            else:
                print("Pre-computing features using CPU...")
                self._precompute_features_cpu()

            # After pre-computation, if using on-disk cache, save metadata
            if self.cache_dir:
                self._save_metadata()

        elif not loaded_from_cache:
            print("Feature pre-computation disabled and no cache loaded - computing on-the-fly")

        print(f"[TIMER] InteractionGraphDatasetLegacy.__init__ total time: {time.time() - init_start_time:.4f} seconds")

    def __del__(self):
        """Ensure the file handle is closed when the dataset object is destroyed."""
        if self.cache_file_handle:
            self.cache_file_handle.close()

    def _save_metadata(self):
        """Saves essential dataset metadata to the cache directory."""
        if not self.cache_dir:
            return

        metadata = {
            'valid_indices': self.valid_indices,
            'precompute_batch_size': self.precompute_batch_size
        }
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved dataset metadata to {self.metadata_path}")

    def _load_molecules_in_parallel(self, paths):
        """Load a list of molecules in parallel and return a dictionary."""
        num_workers = int(os.cpu_count() / 2)
        chunk_cache = {}
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            for i, (path, molecule) in enumerate(pool.imap_unordered(_load_molecule_worker, paths)):
                if molecule is not None:
                    chunk_cache[path] = molecule
        
        return chunk_cache

    def _preload_molecules(self):
        """Pre-load molecules into cache for faster access using multiprocessing."""
        print("Pre-loading molecules into cache using multiprocessing...")
        unique_paths = list(set(self.molecule_paths_1 + self.molecule_paths_2))
        num_paths = len(unique_paths)

        if num_paths == 0:
            print("No molecules to pre-load.")
            return

        # Use a reasonable number of workers, e.g., number of CPU cores, capped to avoid resource exhaustion
        num_workers = int(os.cpu_count() / 2)
        print(f"Using {num_workers} worker processes for molecule caching.")

        with multiprocessing.Pool(processes=num_workers) as pool:
            # Using imap_unordered to start processing as soon as jobs are finished
            # and to populate the cache as results come in.
            for i, (path, molecule) in enumerate(pool.imap_unordered(_load_molecule_worker, unique_paths)):
                if molecule is not None:
                    self.molecule_cache[path] = molecule
                if (i + 1) % 1000 == 0 or (i + 1) == num_paths:
                    print(f"Processed {i + 1}/{num_paths} molecules for caching...")

        print(f"Successfully cached {len(self.molecule_cache)}/{num_paths} molecules")

    def _get_molecule(self, path):
        """Get molecule from cache or load it"""
        if path in self.molecule_cache:
            return self.molecule_cache[path]
        return load_molecule(path)

    def _get_coordinates_tensor(self, molecule):
        """Extract coordinates from molecule as torch tensor"""
        return InteractionGraphBuilder.get_coordinates_from_molecule(molecule)

    def _default_property_pairs(self):
        # Return the default property pairs dictionary
        return OrderedDict([
            # Signed symmetric (3 columns each), 8 properties: 24 columns total
            (('polarized_gasteiger_charges', 'polarized_gasteiger_charges'), 'signed'),                 # Coulombic
            (('gasteiger_charges','gasteiger_charges'), 'signed'),                                      # Coulombic
            (('formal_charge', 'formal_charge'), 'signed'),                                             # Coulombic
            (('is_h_ternary','is_h_ternary'), 'signed'),                                                # LJ
            (('is_h_bond_donor_ternary','is_h_bond_donor_ternary'), 'signed'),                          # H-bond
            (('polarized_hydrophobic_ternary','polarized_hydrophobic_ternary'), 'signed'),              # LJ
            (('is_in_aromatic_ring_ternary','is_in_aromatic_ring_ternary'), 'signed'),                  # Aromatic/pi-stacking
            (('eneg_difference_carbon', 'eneg_difference_carbon'), 'signed'),

            # Unsigned symmetric (1 column each), 4 properties: 4 columns total
            (('electronegativity', 'electronegativity'), 'unsigned'),                                   # Scaling
            (('polarized_vdw_radius', 'polarized_vdw_radius'), 'unsigned'),                             # LJ
            (('vdw_radius', 'vdw_radius'), 'unsigned'),                                                 # LJ
            (('polarizability', 'polarizability'), 'unsigned'),                                         # Scaling

            # Asymmetric (3 columns each), 12 properties: 36 columns total
            (('electronegativity', 'gasteiger_charges'), 'signed'),                            # Coulombic
            (('gasteiger_charges', 'electronegativity'), 'signed'),                            # Coulombic
            (('eneg_difference_carbon', 'polarized_hydrophobic_ternary'), 'signed'),           #
            (('polarized_hydrophobic_ternary', 'eneg_difference_carbon'), 'signed'),           #
            (('is_in_ring_ternary', 'polarized_hydrophobic_ternary'), 'signed'),               # Pi-LJ
            (('polarized_hydrophobic_ternary', 'is_in_ring_ternary'), 'signed'),               # LJ-pi
            (('is_in_aromatic_ring_ternary', 'polarized_hydrophobic_ternary'), 'signed'),      # Pi-LJ
            (('polarized_hydrophobic_ternary', 'is_in_aromatic_ring_ternary'), 'signed'),      # LJ-pi
            (('is_in_aromatic_ring_ternary', 'polarized_gasteiger_charges'), 'signed'),        # Pi-Cation
            (('polarized_gasteiger_charges', 'is_in_aromatic_ring_ternary'), 'signed'),        # Cation-Pi
            (('is_in_aromatic_ring_ternary', 'polarized_vdw_radius'), 'signed'),               # Pi-LJ
            (('polarized_vdw_radius', 'is_in_aromatic_ring_ternary'), 'signed')                # Pi-LJ
        ])

    def _read_molecule_paths(self, ligand_file, protein_file, ligand_protein_pair_file):
        """
        Reads molecule file paths from the specified source.

        Parameters:
            ligand_file (str, optional): Path to CSV file for the first set of molecules.
            protein_file (str, optional): Path to CSV file for the second set of molecules.
            ligand_protein_pair_file (str, optional): Path to CSV file containing paths for both sets of molecules.

        Returns:
            Tuple of lists containing paths for the first and second set of molecules.
        """
        if ligand_protein_pair_file:
            molecule_paths = pd.read_csv(ligand_protein_pair_file, header=None, delimiter=';')
            return molecule_paths[0].tolist(), molecule_paths[1].tolist()
        elif ligand_file and protein_file:
            with open(ligand_file, 'r') as f:
                ligand_paths = [line.strip() for line in f.readlines()]
            with open(protein_file, 'r') as f:
                protein_paths = [line.strip() for line in f.readlines()]
            return ligand_paths, protein_paths
        else:
            raise ValueError("Either a ligand_protein_pair_file or a pair of molecule files must be provided.")


    def __len__(self):
        """
        Returns the number of valid items in the dataset.
        """
        return len(self.valid_indices)

    def normalize_features(self, feature_tensor, mean, std):
        """Normalize the feature tensor using the provided mean and std."""
        original_shape = feature_tensor.shape

        # Completely flatten the tensor to match the saved normalization data
        flattened_samples = feature_tensor.reshape(-1, original_shape[-1] * original_shape[-2])
        
        # Ensure mean and std are on the same device as feature_tensor
        device = feature_tensor.device
        mean = mean.to(device)
        std = std.to(device)
        
        # Reshape mean and std to match flattened_samples shape
        mean = mean.view(1, -1)  # Shape: [1, 4096] for 64 features with 64 bins
        std = std.view(1, -1)    # Shape: [1, 4096] for 64 features with 64 bins
        
        # Handle NaNs in std in-place
        std.nan_to_num_(0.0)
        std[std == 0] = 1

        # Normalize in-place
        flattened_samples -= mean
        flattened_samples.nan_to_num_(0.0)
        flattened_samples /= std

        # Reshape back to original shape
        return flattened_samples.view(original_shape)

    def __getitem__(self, index):
        """
        Get a single item from the dataset - now just a simple lookup!
        """
        start_time = time.time()
        if not self.valid_indices:
            raise RuntimeError("No more valid items in the dataset")

        true_index = self.valid_indices[index % len(self.valid_indices)]
        
        feature_data = None
        # On-disk cache mode
        cache_read_start_time = time.time()
        if self.cache_dir and self.cache_index:
            if true_index in self.cache_index:
                if self.cache_file_handle is None:
                    # Open file handle if not already open (e.g., in a new worker)
                    self.cache_file_handle = open(self.cache_file, 'rb')
                
                offset, length = self.cache_index[true_index]
                self.cache_file_handle.seek(offset)
                data_bytes = self.cache_file_handle.read(length)
                
                buffer = io.BytesIO(data_bytes)
                feature_data = torch.load(buffer, map_location='cpu')

        # In-memory cache mode for legacy or non-precomputed
        elif true_index in self.features_cache:
            feature_data = self.features_cache[true_index]
        # On-the-fly needs to load from disk
        else:
            return self._compute_features_on_the_fly(index, true_index)
        cache_read_time = time.time() - cache_read_start_time

        processing_start_time = time.time()
        if feature_data is not None:
            # Use pre-computed features
            feature_tensor = feature_data['features'].clone()  # Clone to avoid modifying cached version
            
            # Apply normalization if needed
            if self.inference and self.load_normalization_data_pkl is not None:
                mean, std = normalization_utils.load_mean_std(self.load_normalization_data_pkl)
                feature_tensor = self.normalize_features(feature_tensor, mean, std)
            
            # Apply transform if specified
            if self.transform:
                feature_tensor = self.transform(feature_tensor)
            
            # Ensure feature tensor is on CPU for DataLoader compatibility
            if isinstance(feature_tensor, torch.Tensor) and feature_tensor.is_cuda:
                feature_tensor = feature_tensor.cpu()
            
            sample = {"features": feature_tensor, "original_index": true_index}
            
            processing_time = time.time() - processing_start_time
            total_time = time.time() - start_time
            print(f"[TIMER] __getitem__ index {index}: Total={total_time:.6f}s (CacheRead={cache_read_time:.6f}s, Processing={processing_time:.6f}s)")
            return sample
        
        else:
            # Fall back to on-the-fly computation (old behavior)
            return self._compute_features_on_the_fly(index, true_index)

    def _compute_features_on_the_fly(self, index, true_index):
        """
        Compute features on-the-fly. This is a fallback for when pre-computation is disabled or a sample is missing from cache.
        This method computes a single item and does not retry, as that can break dataloader indexing.
        """
        try:
            # Load molecules
            ligand = self._get_molecule(self.molecule_paths_1[true_index])
            protein = self._get_molecule(self.molecule_paths_2[true_index])
            
            if ligand is None or protein is None:
                raise ValueError("One or both molecules failed to load")

            # Build interaction graph (simplified - no GPU batch processing here)
            builder = InteractionGraphBuilder(
                ligand, protein,
                search_method=self.search_method,
                distance_cutoff=self.distance_cutoff,
                reduce_interaction_graph=self.reduce_interaction_graph,
                debug=False
            )

            # Create featurizer
            featurizer = InteractionGraphPropertyDistanceFeaturizerGPU(
                ligand, protein,
                interaction_graph_builder=builder,
                step_size=self.step_size,
                num_distance_bins=self.num_distance_bins,
                property_pairs=self.property_pairs
            )
            feature_tensor = featurizer.feature_tensor

            # Handle normalization
            if self.inference and self.load_normalization_data_pkl is not None:
                mean, std = normalization_utils.load_mean_std(self.load_normalization_data_pkl)
                feature_tensor = self.normalize_features(feature_tensor, mean, std)

            if self.transform:
                feature_tensor = self.transform(feature_tensor)
            
            # Ensure feature tensor is on CPU for DataLoader compatibility
            feature_tensor = feature_tensor.cpu()
                
            sample = {"features": feature_tensor, "original_index": true_index}
            
            return sample

        except Exception as e:
            print(f"FATAL: Error processing item on-the-fly at original index {true_index}: {str(e)}")
            print(f"  Molecule 1: {self.molecule_paths_1[true_index]}")
            print(f"  Molecule 2: {self.molecule_paths_2[true_index]}")
            print("  This error should have been caught during validation. Consider re-running with fresh validation.")
            # The default collate_fn in PyTorch will crash on None. A custom collate_fn would be needed to handle this gracefully.
            # Re-raising the exception to make the error explicit, as this path should ideally not be used.
            raise e

    def _precompute_features_gpu(self):
        """
        Pre-compute all features using GPU batch processing and save to disk if cache_dir is provided.
        """
        precompute_start_time = time.time()
        initial_valid_indices = list(range(len(self.molecule_paths_1)))
        
        # Keep track of successful feature computations
        successful_indices = []
        
        print(f"Processing {len(initial_valid_indices)} samples in chunks of {self.precompute_batch_size}")

        new_cache_index = {}
        # Ensure the cache file is cleared before starting
        if self.cache_dir:
            with open(self.cache_file, 'wb'): # as f:
                pass

        for i in range(0, len(initial_valid_indices), self.precompute_batch_size):
            chunk_start_time = time.time()
            chunk_indices = initial_valid_indices[i:i + self.precompute_batch_size]
            chunk_num = i // self.precompute_batch_size
            print(f"Processing chunk {chunk_num + 1}/{(len(initial_valid_indices) - 1) // self.precompute_batch_size + 1}")

            # Load molecules for the current chunk
            paths1 = [self.molecule_paths_1[idx] for idx in chunk_indices]
            paths2 = [self.molecule_paths_2[idx] for idx in chunk_indices]
            unique_paths = list(set(paths1 + paths2))
            
            mol_load_start = time.time()
            print(f"  Loading {len(unique_paths)} unique molecules for this chunk...")
            chunk_molecule_cache = self._load_molecules_in_parallel(unique_paths)
            print(f"  ...done loading molecules in {time.time() - mol_load_start:.4f}s.")

            molecules1_chunk = []
            molecules2_chunk = []
            valid_chunk_indices = [] # Original indices from the full dataset

            for idx in chunk_indices:
                try:
                    mol1 = chunk_molecule_cache.get(self.molecule_paths_1[idx])
                    mol2 = chunk_molecule_cache.get(self.molecule_paths_2[idx])
                    if mol1 is not None and mol2 is not None:
                        molecules1_chunk.append(mol1)
                        molecules2_chunk.append(mol2)
                        valid_chunk_indices.append(idx)
                    else:
                        print(f"Failed to load molecules for sample {idx}")
                        self.skipped_indices.append(idx)
                except Exception as e:
                    print(f"Error accessing molecules for sample {idx}: {e}")
                    self.skipped_indices.append(idx)
            
            if not molecules1_chunk:
                print("No valid molecules in this chunk, skipping.")
                # Explicitly clean up before the next iteration
                del chunk_molecule_cache
                gc.collect()
                continue

            # Extract coordinates
            coords1_batch = [self._get_coordinates_tensor(mol) for mol in molecules1_chunk]
            coords2_batch = [self._get_coordinates_tensor(mol) for mol in molecules2_chunk]

            # Temp builder for this chunk
            temp_builder = InteractionGraphBuilder(
                molecules1_chunk[0], molecules2_chunk[0], # Dummy molecules
                search_method='cdist',
                distance_cutoff=self.distance_cutoff,
                reduce_interaction_graph=False,
                debug=False,
                device=self.device
            )

            # Build interaction graphs for the chunk
            graph_build_start = time.time()
            interaction_graphs, distance_matrices = temp_builder.build_batch_interaction_graphs(
                coords1_batch, coords2_batch, device=self.device
            )
            if self.gpu_available:
                torch.cuda.synchronize()
            print(f"  Built interaction graphs in {time.time() - graph_build_start:.4f}s.")
            
            # Reduce graph if necessary
            if self.reduce_interaction_graph:
                reduce_start = time.time()
                reduced_graphs, interacting_atoms1_batch, interacting_atoms2_batch, reduced_distances = \
                    temp_builder.reduce_interaction_graphs_batch(interaction_graphs, distance_matrices)
                interaction_graphs = reduced_graphs
                distance_matrices = reduced_distances
                if self.gpu_available:
                    torch.cuda.synchronize()
                print(f"  Reduced graphs in {time.time() - reduce_start:.4f}s.")
            else:
                interacting_atoms1_batch = [list(range(ig.shape[0])) for ig in interaction_graphs]
                interacting_atoms2_batch = [list(range(ig.shape[1])) for ig in interaction_graphs]

            # Compute and cache/save features for each sample in the chunk
            feature_compute_start = time.time()
            chunk_features = {}
            for j, (mol1, mol2, interaction_graph, distances, atoms1, atoms2) in enumerate(
                zip(molecules1_chunk, molecules2_chunk, interaction_graphs, distance_matrices,
                    interacting_atoms1_batch, interacting_atoms2_batch)):

                original_index = valid_chunk_indices[j]
                
                try:
                    # Build features using chemical properties
                    temp_builder_for_features = type('TempBuilder', (), {
                        'atom_pair_distances': distances,
                        'interaction_graph': interaction_graph,
                        'interacting_atoms1': atoms1,
                        'interacting_atoms2': atoms2
                    })()
                    
                    featurizer = InteractionGraphPropertyDistanceFeaturizerGPU(
                        mol1, mol2,
                        interaction_graph_builder=temp_builder_for_features,
                        step_size=self.step_size,
                        num_distance_bins=self.num_distance_bins,
                        property_pairs=self.property_pairs,
                        device=self.device
                    )
                    
                    feature_tensor = featurizer.feature_tensor.cpu()
                    cache_dict = {'features': feature_tensor}

                    # Add to chunk's feature dictionary
                    chunk_features[original_index] = cache_dict
                    successful_indices.append(original_index)
                    
                except Exception as e:
                    print(f"Error computing features for sample {original_index}: {e}")
                    self.skipped_indices.append(original_index)
            
            if self.gpu_available:
                torch.cuda.synchronize()
            print(f"  Computed features for chunk in {time.time() - feature_compute_start:.4f}s.")

            # Save features for the chunk to the single file or in-memory cache
            save_chunk_start = time.time()
            if self.cache_dir and chunk_features:
                with open(self.cache_file, 'ab') as cache_file_handle:
                    for original_index, cache_dict in chunk_features.items():
                        buffer = io.BytesIO()
                        torch.save(cache_dict, buffer)
                        serialized_data = buffer.getvalue()
                        
                        offset = cache_file_handle.tell()
                        length = len(serialized_data)
                        
                        cache_file_handle.write(serialized_data)
                        new_cache_index[original_index] = (offset, length)
            else:
                self.features_cache.update(chunk_features)
            
            print(f"  Saved chunk features in {time.time() - save_chunk_start:.4f}s.")

            # Explicitly release memory from the completed chunk
            del chunk_molecule_cache
            del molecules1_chunk
            del molecules2_chunk
            del coords1_batch
            del coords2_batch
            del interaction_graphs
            del distance_matrices
            del interacting_atoms1_batch
            del interacting_atoms2_batch
            del chunk_features
            try:
                # These are created conditionally, so they might not exist
                del reduced_graphs
                del reduced_distances
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  Chunk processing time: {time.time() - chunk_start_time:.4f}s.")
        
        # Finalize and save the index
        if self.cache_dir:
            self.cache_index = new_cache_index
            with open(self.cache_index_path, 'wb') as f_index:
                pickle.dump(self.cache_index, f_index)
            print(f"Saved feature cache with {len(self.cache_index)} samples to {self.cache_file}")

        # Finalize
        self.valid_indices = sorted(successful_indices)

        print(f"Successfully pre-computed features for {len(self.valid_indices)} samples")
        print(f"Skipped {len(self.skipped_indices)} samples due to errors")
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        print(f"[TIMER] _precompute_features_gpu total time: {time.time() - precompute_start_time:.4f} seconds")

    def _precompute_features_cpu(self):
        """
        Pre-compute all features using CPU (fallback method) and save to disk if cache_dir is provided.
        """
        precompute_start_time = time.time()
        print("Pre-computing features using CPU...")
        
        successful_indices = []
        
        new_cache_index = {}
        # Ensure the cache file is cleared before starting
        if self.cache_dir:
            with open(self.cache_file, 'wb'): # as f:
                pass

        # Process in chunks to be memory-efficient, even on CPU
        for i in range(0, len(self.valid_indices), self.precompute_batch_size):
            chunk_indices = self.valid_indices[i:i + self.precompute_batch_size]
            chunk_num = i // self.precompute_batch_size
            print(f"Processing chunk {chunk_num + 1}/{(len(self.valid_indices) - 1) // self.precompute_batch_size + 1}")
            
            paths1 = [self.molecule_paths_1[idx] for idx in chunk_indices]
            paths2 = [self.molecule_paths_2[idx] for idx in chunk_indices]
            unique_paths = list(set(paths1 + paths2))

            mol_load_start = time.time()
            print(f"  Loading {len(unique_paths)} unique molecules for this chunk...")
            chunk_molecule_cache = self._load_molecules_in_parallel(unique_paths)
            print(f"  ...done loading molecules in {time.time() - mol_load_start:.4f}s.")
            
            feature_compute_start = time.time()
            chunk_features = {}
            for idx in chunk_indices:
                try:
                    # Load molecules from chunk cache
                    mol1 = chunk_molecule_cache.get(self.molecule_paths_1[idx])
                    mol2 = chunk_molecule_cache.get(self.molecule_paths_2[idx])
                    
                    if mol1 is None or mol2 is None:
                        raise ValueError("Failed to load molecules from chunk cache")
                    
                    # Build interaction graph
                    builder = InteractionGraphBuilder(
                        mol1, mol2,
                        search_method=self.search_method,
                        distance_cutoff=self.distance_cutoff,
                        reduce_interaction_graph=self.reduce_interaction_graph,
                        debug=False
                    )
                    
                    cache_dict = {}
                    # Pre-compute the final feature tensor
                    featurizer = InteractionGraphPropertyDistanceFeaturizerGPU(
                        mol1, mol2,
                        interaction_graph_builder=builder,
                        step_size=self.step_size,
                        num_distance_bins=self.num_distance_bins,
                        property_pairs=self.property_pairs
                    )
                    cache_dict['features'] = featurizer.feature_tensor

                    # Add to chunk's feature dictionary
                    chunk_features[idx] = cache_dict
                    successful_indices.append(idx)

                except Exception as e:
                    print(f"Error pre-computing features for sample {idx}: {e}")
                    self.skipped_indices.append(idx)
            
            print(f"  Computed features for chunk in {time.time() - feature_compute_start:.4f}s.")

            # Save features for the chunk to the single file or in-memory cache
            save_chunk_start = time.time()
            if self.cache_dir and chunk_features:
                with open(self.cache_file, 'ab') as cache_file_handle:
                    for original_index, cache_dict in chunk_features.items():
                        buffer = io.BytesIO()
                        torch.save(cache_dict, buffer)
                        serialized_data = buffer.getvalue()
                        
                        offset = cache_file_handle.tell()
                        length = len(serialized_data)
                        
                        cache_file_handle.write(serialized_data)
                        new_cache_index[original_index] = (offset, length)
            else:
                self.features_cache.update(chunk_features)
            
            print(f"  Saved chunk features in {time.time() - save_chunk_start:.4f}s.")

            # Explicitly release memory from the completed chunk
            del chunk_molecule_cache
            del chunk_features
            gc.collect()
        
        # Finalize and save the index
        if self.cache_dir:
            self.cache_index = new_cache_index
            with open(self.cache_index_path, 'wb') as f_index:
                pickle.dump(self.cache_index, f_index)
            print(f"Saved feature cache with {len(self.cache_index)} samples to {self.cache_file}")

        self.valid_indices = sorted(successful_indices)

        print(f"Successfully pre-computed features for {len(self.valid_indices)} samples")
        print(f"[TIMER] _precompute_features_cpu total time: {time.time() - precompute_start_time:.4f} seconds")


    def num_feature_columns(self):
        """
        Determines the number of feature columns based on the binning scheme.

        Returns:
            int: Number of feature columns.
        """
        # Count the number of columns based on binning scheme
        bin_contributions = {
            'unsigned': 1,
            'signed': 3,                # -/- , +/+ , -/+
            'signed_directional': 4     # -/- , +/+ , -/+ , +/-
        }
        n_feature_cols = 0
        for binning_scheme in self.property_pairs.values():
            n_feature_cols += bin_contributions.get(binning_scheme, np.nan)
        if np.isnan(n_feature_cols):
            raise ValueError(f"Unsupported binning scheme in list: {self.property_pairs.values()}")

        return n_feature_cols

    def num_features(self):
        """
        Calculates the total number of features in the dataset (per interacting atom pair).

        Returns:
            int: Total number of features.
        """
        return self.num_feature_columns() * self.num_distance_bins  # This is technically per interaction because it has not been pooled

    def num_results(self):
        """
        Returns 0 since this is an inference-only dataset with no result labels.

        Returns:
            int: Always returns 0.
        """
        return 0


    def shuffle(self):
        """
        Shuffle the dataset by randomly permuting the samples.
        Handles both data loaded from files and from a pre-computed cache.
        """
        # Case 1: Data loaded from original files, shuffle molecule paths
        if hasattr(self, 'molecule_paths_1') and self.molecule_paths_1:
            num_samples = len(self.molecule_paths_1)
            permutation = torch.randperm(num_samples)

            # Shuffle molecule paths
            self.molecule_paths_1 = [self.molecule_paths_1[i] for i in permutation]
            self.molecule_paths_2 = [self.molecule_paths_2[i] for i in permutation]
        
        # Case 2: Data loaded from cache, shuffle the valid indices
        else:
            import random
            random.shuffle(self.valid_indices)
            print("Shuffled valid indices for cached dataset.")

    def get_skipped_indices(self):
        """
        Get the list of indices that were skipped due to errors.

        Returns:
            list: List of skipped indices.
        """
        return self.skipped_indices
    def has_valid_items(self):
        """
        Check if there are any valid items left in the dataset.

        Returns:
            bool: True if there are valid items, False otherwise.
        """
        return len(self.valid_indices) > 0

    def create_gpu_batch_features(self, molecules1_batch, molecules2_batch, 
                                interaction_graphs_batch, distances_batch, 
                                interacting_atoms1_batch, interacting_atoms2_batch):
        """
        Create features for a batch of samples using GPU acceleration
        
        Args:
            molecules1_batch, molecules2_batch: Lists of molecules
            interaction_graphs_batch: List of interaction graph tensors (on GPU)
            distances_batch: List of distance matrices (on GPU) 
            interacting_atoms1_batch, interacting_atoms2_batch: Lists of interacting atom indices
            
        Returns:
            Batched feature tensor
        """
        from modules.features.compute_properties import unified_compute_atomic_properties
        
        batch_features = []
        
        for i, (mol1, mol2, interaction_graph, distances, atoms1, atoms2) in enumerate(
            zip(molecules1_batch, molecules2_batch, interaction_graphs_batch, 
                distances_batch, interacting_atoms1_batch, interacting_atoms2_batch)):
            
            # Move to CPU for property computation (this is still needed)
            distances_cpu = distances.cpu()
            interaction_graph_cpu = interaction_graph.cpu()
            
            # Compute properties on CPU (this part is hard to GPU-accelerate due to RDKit)
            keys = list(self.property_pairs.keys())
            mol1_properties = unified_compute_atomic_properties(
                mol1, [k[0] for k in keys], [mol2], self.step_size * self.num_distance_bins
            )
            mol2_properties = unified_compute_atomic_properties(
                mol2, [k[1] for k in keys], [mol1], self.step_size * self.num_distance_bins
            )
            
            # GPU-accelerated histogram computation
            feature_tensor = self._gpu_compute_histogram_features(
                mol1_properties, mol2_properties, distances_cpu, interaction_graph_cpu,
                atoms1, atoms2
            )
            
            batch_features.append(feature_tensor)
        
        return torch.stack(batch_features)
    
    def _gpu_compute_histogram_features(self, mol1_properties, mol2_properties, 
                                      distances, interaction_graph, atoms1, atoms2):
        """
        GPU-accelerated histogram feature computation
        """
        # Initialize feature bins
        binning_factors = {'unsigned': 1, 'signed': 3, 'signed_directional': 4}
        total_bins = sum(binning_factors[scheme] for scheme in self.property_pairs.values())
        feature_bins = torch.zeros(self.num_distance_bins, total_bins, device=self.device)
        
        # Move data to GPU
        distances = distances.to(self.device)
        interaction_graph = interaction_graph.to(self.device)
        
        # Get interaction indices
        interaction_indices = torch.nonzero(interaction_graph, as_tuple=False)
        
        if len(interaction_indices) == 0:
            return feature_bins
        
        # Compute distance bins
        distance_bins = torch.round(distances[interaction_indices[:, 0], interaction_indices[:, 1]] / self.step_size).long()
        valid_mask = distance_bins < self.num_distance_bins
        
        interaction_indices = interaction_indices[valid_mask]
        distance_bins = distance_bins[valid_mask]
        
        if len(interaction_indices) == 0:
            return feature_bins
        
        # Process each property pair
        bin_offset = 0
        for (prop1_name, prop2_name), binning_scheme in self.property_pairs.items():
            # Get property values for interacting atoms
            mol1_vals = torch.tensor([mol1_properties[prop1_name][atoms1[i]] 
                                    for i in interaction_indices[:, 0]], device=self.device)
            mol2_vals = torch.tensor([mol2_properties[prop2_name][atoms2[i]] 
                                    for i in interaction_indices[:, 1]], device=self.device)
            
            products = mol1_vals * mol2_vals
            nonzero_mask = products != 0.0
            
            if not nonzero_mask.any():
                bin_offset += binning_factors[binning_scheme]
                continue
            
            # Filter to non-zero products
            filtered_bins = distance_bins[nonzero_mask]
            filtered_products = products[nonzero_mask]
            filtered_mol1_vals = mol1_vals[nonzero_mask]
            filtered_mol2_vals = mol2_vals[nonzero_mask]
            
            # Binning logic
            if binning_scheme == 'unsigned':
                # Simple accumulation
                feature_bins.index_add_(0, filtered_bins, 
                                      filtered_products.unsqueeze(1).expand(-1, 1))
                
            elif binning_scheme == 'signed':
                products_abs = torch.abs(filtered_products)
                
                # Create masks for different sign combinations  
                neg_neg = (filtered_mol1_vals < 0) & (filtered_mol2_vals < 0)
                pos_pos = (filtered_mol1_vals >= 0) & (filtered_mol2_vals >= 0)
                opp_sign = (filtered_mol1_vals * filtered_mol2_vals < 0)
                
                # Accumulate into different bins
                if neg_neg.any():
                    indices = filtered_bins[neg_neg]
                    values = products_abs[neg_neg].unsqueeze(1)
                    feature_bins[:, bin_offset:bin_offset+1].index_add_(0, indices, values)
                
                if pos_pos.any():
                    indices = filtered_bins[pos_pos]
                    values = products_abs[pos_pos].unsqueeze(1)
                    feature_bins[:, bin_offset+1:bin_offset+2].index_add_(0, indices, values)
                
                if opp_sign.any():
                    indices = filtered_bins[opp_sign]
                    values = products_abs[opp_sign].unsqueeze(1)
                    feature_bins[:, bin_offset+2:bin_offset+3].index_add_(0, indices, values)
            
            # Add similar logic for signed_directional...
            bin_offset += binning_factors[binning_scheme]
        
        return feature_bins

    # Add a custom collate function
    @staticmethod
    def gpu_collate_fn(batch):
        """
        Custom collate function for GPU batch processing
        """
        features = torch.stack([item['features'] for item in batch])
        original_indices = [item['original_index'] for item in batch]
        
        return {
            'features': features,
            'original_index': original_indices
        }

    # Update the DataLoaderFactory to use our custom collate function
    def get_dataloader(self, batch_size=32, shuffle=False, num_workers=0):
        """
        Create a DataLoader with GPU batch processing
        """
        from torch.utils.data import DataLoader
        
        if self.gpu_available:
            # Use custom collate function for GPU batching
            return DataLoader(
                self, 
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self.gpu_collate_fn,
                pin_memory=True
            )
        else:
            # Fall back to default
            return DataLoader(
                self,
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=num_workers
            )


