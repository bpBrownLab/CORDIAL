#!/usr/bin/env python

# General imports
from collections import OrderedDict

# PyTorch imports
from torch.utils.data.distributed import DistributedSampler

# Project imports
from modules.datasets.data_loader_factory import DataLoaderFactory
from modules.datasets.interaction_graph_dataset_legacy import InteractionGraphDatasetLegacy

class DatasetHandler:
    def __init__(self,
                 num_workers=0,
                 load_dataset=None, load_and_combine_datasets=None, flatten_loaded_dataset=False, dataset_impurity_procedure='none',
                 ligand_file=None, protein_file=None, ligand_protein_pair_file=None,
                 sampler_type='none', world_size=None, rank=None,
                 oversample_target_ratio=None,
                 dataset_type='csv',
                 first_column_row_id=False, num_result_classes=8,
                 batch_method='mini-batch', batch_size=32,
                 inference=False,
                 load_normalization_data_pkl=None, save_normalization_data_pkl=None, skip_normalization=False,
                 search_method='cdist', distance_cutoff=16.0, step_size=0.25, num_distance_bins=64,
                 reduce_interaction_graph=False, filter_obstructions=False, obstruction_tolerance=0.0,
                 property_pairs=None, exclude_feature_columns=None,
                 pin_memory=False,
                 cache_dir="./",
                 precompute_features=False,
                 device=None
                 ):

        # Number of workers for data loader
        self.num_workers = num_workers

        # Arbitrary pre-generated dataset
        self.load_dataset = load_dataset                              # Loaded as a GenericDataset and normalized by mean and std
        self.load_and_combine_datasets = load_and_combine_datasets    # Loaded as GenericDatasets and normalized by mean and std
        self.flatten_loaded_dataset = flatten_loaded_dataset          # Flatten the loaded GenericDataset
        self.dataset_impurity_procedure = dataset_impurity_procedure  # How to handle inf/nan values in dataset samples
        self.exclude_feature_columns = exclude_feature_columns        # Remove feature columns during dataset loading

        # Forms of input molecules
        self.ligand_file = ligand_file                                # SDF ligand molecules
        self.protein_file = protein_file                              # PDB protein molecules
        self.ligand_protein_pair_file = ligand_protein_pair_file      # Comma-separated SDF,PDB file pairs

        # Forms of result labels
        self.num_result_classes = num_result_classes                  # Number of final columns interpreted as result labels
        
        # Dataset details
        self.sampler_type = sampler_type                              # Type of sampler to use
        self.world_size = world_size                                  # Total number of processes
        self.rank = rank                                              # Rank of the current process
        self.dataset_type = dataset_type                              # Determines how input files will be parsed
        
        self.search_method = search_method                            # Interaction graph dataset
        self.distance_cutoff = distance_cutoff                        # Interaction graph dataset
        self.step_size = step_size                                    # Interaction graph dataset
        self.num_distance_bins = num_distance_bins                    # Interaction graph dataset
        self.reduce_interaction_graph = reduce_interaction_graph      # Remove atoms from the interaction graph that do not form any interactions
        self.filter_obstructions = filter_obstructions                # Do not count obstructed atoms toward interactions
        self.obstruction_tolerance = obstruction_tolerance            # Tolerance level governing obstruction filtering
        self.precompute_features = precompute_features

        # Property pairs
        self.property_pairs = property_pairs if property_pairs is not None else OrderedDict([
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

        # Training, validation, testing, and inference
        self.batch_method = batch_method                              # How to match samples
        self.batch_size = batch_size                                  # Batch size for mini-batch method
        self.inference = inference                                    # Run inference with a model

        self.load_normalization_data_pkl = load_normalization_data_pkl  # File to save/load training normalization data to/from
        self.skip_normalization = skip_normalization                    # Skip normalization

        # Useful for GPU training
        self.device = device
        self.pin_memory = pin_memory                                    # Pin CPU for GPU training

        # Cache directory
        self.cache_dir = cache_dir                                      # Cache directory

    """
    A class for handling dataset creation and DataLoader setup.

    Args:
        input_molecules (str): Path to the input molecules file.
        shuffle_input_molecules (bool): Whether to shuffle input molecules.
        dataset_type (str): Type of dataset ("csv" or "smiles").
        splits (list): List of three fractions for train, validation, and test splits.
        batch_method (str): Batch method ("full-batch", "online-learning", or "mini-batch").
        batch_size (int): Batch size.

    Attributes:
        input_molecules (str): Path to the input molecules file.    
        shuffle_input_molecules (bool): Whether to shuffle input molecules.
        dataset_type (str): Type of dataset ("csv" or "smiles").
        splits (list): List of three fractions for train, validation, and test splits.
        batch_method (str): Batch method ("full-batch", "online-learning", or "mini-batch").
        batch_size (int): Batch size.

    Methods:
        generate_datasets(): Generate training, validation, and test datasets.
        create_data_loaders(dataset, train_set, validation_set, test_set, inference): Create DataLoader objects for training, validation, and test sets.

    """

    # TODO: needs to be updated for new interaction graph
    def remove_feature_columns(self, features):
        assert self.exclude_feature_columns is not None, "Cannot exclude feature columns if no feature columns specified!"
        n_feature_columns = 0
        if self.dataset_type == "interaction_graph_legacy":
            for value in self.property_pairs.values():
                if value == "unsigned":
                    n_feature_columns += 1
                elif value == "signed":
                    n_feature_columns += 3
                elif value == "signed_directional":
                    n_feature_columns += 4
            include_feature_columns = [i for i in range(n_feature_columns) if i not in self.exclude_feature_columns]
            features = features[:, :, include_feature_columns]  # [batch_size, n_distance_bins, n_feature_columns]
        return features

    def generate_datasets(self):
        """
        Generate training, validation, and test datasets.

        Returns:
            Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
                Full, training, validation, and test datasets.

        """
        # Load a dataset that was previously created using PyTorch and processed via a DataLoader
        if self.load_dataset is not None or self.load_and_combine_datasets is not None:
            dataset = self.load_datasets()

        elif self.dataset_type == "interaction_graph_legacy":
            dataset = InteractionGraphDatasetLegacy(
                ligand_file=self.ligand_file, 
                protein_file=self.protein_file, 
                ligand_protein_pair_file=self.ligand_protein_pair_file, 
                property_pairs=self.property_pairs,
                transform=None,
                search_method=self.search_method, 
                distance_cutoff=self.distance_cutoff,
                step_size=self.step_size,
                num_distance_bins=self.num_distance_bins,
                reduce_interaction_graph=self.reduce_interaction_graph,
                inference=self.inference,
                load_normalization_data_pkl=self.load_normalization_data_pkl,
                cache_dir=self.cache_dir,
                precompute_features=self.precompute_features,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        return dataset

    def create_data_loaders(self, dataset=None, train_set=None, validation_set=None, test_set=None, inference=False):
        """
        Create DataLoader objects for training, validation, and test sets.

        Args:
            dataset (torch.utils.data.Dataset): Dataset.
            train_set (torch.utils.data.Dataset): Training dataset.
            validation_set (torch.utils.data.Dataset): Validation dataset.
            test_set (torch.utils.data.Dataset): Test dataset.
            inference (bool): True to create a dataloader from the full dataset

        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
                DataLoader objects for training, validation, and test sets.

        """
        # Generate data loaders
        if self.load_dataset is None and self.load_and_combine_datasets is None:
            if inference:
                # Create distributed sampler for DDP inference if needed
                sampler = None
                if self.sampler_type == "distributed_sampler":
                    sampler = DistributedSampler(dataset, num_replicas=self.world_size, 
                                               rank=self.rank, shuffle=False)
                
                dataset_loader = DataLoaderFactory.create_data_loader(
                    dataset=dataset, alias=self.dataset_type,
                    batch_method=self.batch_method, batch_size=self.batch_size,
                    sampler=sampler,
                    shuffle=False, pin_memory=self.pin_memory,
                    onehot_to_class_indices=True,
                    num_workers=self.num_workers
                )
                return dataset_loader
            
   
        # TODO: dumb that here we have to use the alias 'none' explicitly instead of letting the DataLoaderFactory do the work
        # This is because the DataLoaderFactory uses the alias to determine what kind of dataset to generate, but if we use the 
        # load_dataset or load_and_combine_datasets flags, we we automatically create a GenericDataset, but it is important
        # to note the underlying structure of the GenericDataset, such as whether it was loaded from an InteractionGraphDataset versus
        # a CSVDataset, etc. Should probably update all dataset classes to inherit from a common abstract class or interface that
        # requires an alias, such that specific datasets have specific aliases but generic datasets can accept one of several aliases.
        # Then if we created a generic dataset, it could still have a dataset_type attribute that specifies the underlying structure.
        else:
            print("Creating data loaders for GenericDataset")
            if inference:
                # Create distributed sampler for DDP inference if needed
                sampler = None
                if self.sampler_type == "distributed_sampler":
                    sampler = DistributedSampler(dataset, num_replicas=self.world_size, 
                                               rank=self.rank, shuffle=False)
                
                dataset_loader = DataLoaderFactory.create_data_loader(
                    dataset=dataset, alias='none',
                    batch_method=self.batch_method, batch_size=self.batch_size,
                    sampler=sampler,
                    shuffle=False, pin_memory=self.pin_memory,
                    num_workers=self.num_workers
                )
                return dataset_loader
