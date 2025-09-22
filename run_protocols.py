#!/usr/bin/env python

"""
Run CORDIAL protein-ligand binding affinity prediction.

Author: Benjamin P. Brown, MD, PhD
Email: benjamin.p.brown@vanderbilt.edu
Date: 05/06/2024

"""
# General imports
import numpy as np
import datetime

# PyTorch imports
import torch
import torch.nn as nn

# Project imports
from modules.architectures.model_initializer import ModelInitializer
from modules.datasets.dataset_handler import DatasetHandler
from modules.processes.inference import Inference
from utils.arg_parser_utils import MasterArgumentParser
from utils.logger_utils import log_memory_usage

def init():
    """
    Initialize the script by parsing command-line arguments and setting up the distributed environment.
    This version is capable of running on both GPU and CPU environments.

    Returns:
        tuple: A tuple containing the parsed arguments and the selected device.

    """
    # Initialize arguments
    args_parser = MasterArgumentParser()
    args = args_parser.parse_args()

    if torch.cuda.is_available() and args.device != 'cpu':
        num_gpus = torch.cuda.device_count()

        # manual override
        if args.device is not None:
            device = torch.device(f'cuda:{int(args.device)}')

        # auto based on GPU availability
        elif num_gpus > 1:
            # Use NCCL for GPU if available
            timeout = datetime.timedelta(seconds=1800)  # 30 minutes timeout
            torch.distributed.init_process_group(backend='nccl', timeout=timeout)
            device = torch.device(f"cuda:{torch.distributed.get_rank()}")
            torch.cuda.set_device(device)

        # Default
        else:
            device = torch.device('cuda')  # DP
    else:
        # Use Gloo or MPI for CPU
        num_gpus = 0
        # torch.distributed.init_process_group(backend='gloo')  # 'mpi' could also be used if MPI is configured
        device = torch.device('cpu')

    rank = torch.distributed.get_rank() if num_gpus > 1 and args.device is None else None  # Set to None for DP
    world_size = torch.distributed.get_world_size() if num_gpus > 1 and args.device is None else None  # Set to None for DP

    print(f"Running with world size {world_size}")
    print(f"Running on rank {rank}, using device: {device}")

    # Require at least one protocol
    assert (
            args.inference or
            args.dry_run
    ), \
        "Must specify at least one protocol!"

    # Control reproducibility
    if args.random_seed is not None:
        print("Random seed: " + str(args.random_seed))
        torch.manual_seed(int(args.random_seed))
        np.random.seed(int(args.random_seed))
    else:
        print("Warning: no fixed random seed specified; results will not be fully reproducible.")

    # Inference-only pipeline requires a model to be loaded from file
    if args.inference:
        assert args.load_model is not None, "Must specify which model to load from file!"

    # return args, device
    return args, rank, world_size, device

def create_dataset(args, device, world_size=None, rank=None):
    """
    Create a dataset based on the provided command-line arguments.

    Args:
        args: Parsed command-line arguments.
        world_size: Total number of processes involved.
        rank: The rank of the current process in the distributed setup.
    Returns:
        tuple: A tuple containing the generated dataset and data loader for inference.

    """

    # Distributed setup
    if world_size is not None:
        if world_size > 1 and args.sampler_type == "none":          
            sampler_type = "distributed_sampler"
        else:
            sampler_type = "none"
    else:
        sampler_type = "none"

    # Create the dataset
    dataset_handler = DatasetHandler(
        num_workers=args.num_workers,
        ligand_file=args.input_ligand_file,
        protein_file=args.input_protein_file,
        ligand_protein_pair_file=args.input_ligand_protein_pair_file,
        world_size=world_size,
        rank=rank,
        sampler_type=sampler_type,
        dataset_type=args.dataset_type,
        num_result_classes=int(args.num_result_classes),
        batch_method=args.batch_method,
        batch_size=int(args.batch_size),
        inference=bool(args.inference),
        load_normalization_data_pkl=args.load_normalization_data_pkl,
        skip_normalization=args.skip_normalization,
        search_method=args.search_method,
        distance_cutoff=args.distance_cutoff,
        step_size=args.step_size,
        num_distance_bins=args.num_distance_bins,
        reduce_interaction_graph=args.reduce_interaction_graph,
        filter_obstructions=True if args.filter_obstructions else False,
        obstruction_tolerance=args.obstruction_tolerance,
        property_pairs=None,
        exclude_feature_columns=args.exclude_feature_columns,
        cache_dir=args.cache_dir,
        precompute_features=args.precompute_features,
        device=device,
        pin_memory=True if device != 'cpu' else False  # Pin memory for GPU jobs
    )
    dataset = dataset_handler.generate_datasets()
    print("Number of features per sample: " + str(dataset.num_features()))
    print("Number of feature columns per sample: " + str(dataset.num_feature_columns()))
    print("Number of results per sample: " + str(dataset.num_results()))

    if args.inference:
        inference_loader = dataset_handler.create_data_loaders(dataset=dataset, inference=True)

    return dataset, inference_loader


def create_model(args, device, rank, world_size, dataset, force_data_parallel=False):
    """
    Create and initialize a model with support for DistributedDataParallel.

    Args:
        device:
        force_data_parallel:
        args: Parsed command-line arguments.
        rank: The rank of the current process in the distributed setup.
        world_size: Total number of processes involved.
        dataset: Generated dataset used for model initialization.

    Returns:
        torch.nn.Module: The initialized model, ready for distributed inference.
    """
    # Initialize the model architecture
    model_initializer = ModelInitializer(args=args, dataset=dataset, model_str=args.model_type)
    model = model_initializer.model
    print(f"Model Architecture:\n{model}")

    # Load model from state dict
    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}")

        # Ensure model is loaded to the correct device
        state_dict = torch.load(args.load_model, map_location=device, weights_only=True)

        # Check if the model was trained with DistributedDataParallel (DDP)
        is_ddp = all(key.startswith('module.') for key in state_dict.keys())

        if is_ddp:
            print("DDP-trained model detected. Stripping 'module.' prefix from keys.")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        # Load the state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model weights loaded successfully.")
        except RuntimeError as e:
            print(f"Warning: A RuntimeError occurred during model loading, which may be expected if a component was intentionally changed: {e}")
            print("Attempting to load with strict=False.")
            model.load_state_dict(state_dict, strict=False)


    if torch.cuda.is_available() and args.device != 'cpu':
        if force_data_parallel:
            print("Using DataParallel.")  # Not recommended
            # Move model to the first device
            model = model.to(device)
            model = nn.DataParallel(model)
        elif world_size is not None and world_size > 1:
            # Move model to the correct device and wrap with DDP
            print(f"Adjusting the model for {world_size} processes.")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(f"cuda:{rank}")
            print(f"Using DistributedDataParallel for {world_size} processes.")
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        else:
            print("Using GPU. No parallelization applied.")
            model = model.to(device)
    else:
        print("Using CPU. No parallelization applied.")
        model = model.to("cpu")

    return model

# TODO: need to automatically normalize data the same way it is normalized during training
def run_inference(model, inference_loader, device, parity=None, model_type='mlp', sample_indices=None):
    """
    Run a model prospectively on a dataset.

    Args:
        model: The PyTorch model to be used for inference.
        inference_loader: The DataLoader for the inference dataset.
        device: The device (CPU or GPU) for running inference.
        sample_indices: Specific sample indices to process. Can be a single index, list of indices, or range. Defaults to None (process all samples).
    """
    # Inference
    inference_instance = Inference(model=model, dataset_loader=inference_loader, device=device,
                                 parity=parity, model_type=model_type, 
                                 sample_indices=sample_indices)
    inference_instance.run_inference()
    return inference_instance

def main():
    """
    Main function for running cheminformatics machine learning model protocols.

    """
    # Initialize protocol(s)
    args, rank, world_size, device = init()

    # Generate dataset
    log_memory_usage("run_protocols.py - main - initial")
    dataset, inference_loader = create_dataset(args, device, world_size, rank)
    log_memory_usage("run_protocols.py - main - after create_dataset")

    if args.precompute_features:
        # The pre-computation is handled by the dataset constructor.
        # Once the dataset is created, the cache is built. We can then exit.
        print("Feature cache generated successfully.")

    else:
        # Instantiate the model
        model = create_model(args, device, rank, world_size, dataset)

        # Inference
        if args.inference and not args.dry_run:
            print("Running inference...")
            run_inference(
                model,
                inference_loader,
                device,
                parity=[float(x) for x in args.parity],
                model_type=args.model_type,
                sample_indices=args.inference_sample_indices,
            )

    # End
    print("Protocols complete.")
    return 0


# Execute main() when run from command line
if __name__ == '__main__':
    main()
