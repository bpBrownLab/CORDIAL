#!/usr/bin/env python

from argparse import ArgumentParser

class MasterArgumentParser:
    """Master argument parser for the cheminformatics ML toolkit."""

    def __init__(self):
        self.parser = ArgumentParser(description="Command line arguments for CORDIAL inference")

        # Protocols
        self.protocols_group = self.parser.add_argument_group('Protocols')
        self.protocols_group.add_argument("--inference", dest="inference", required=False, default=False,
                                          action='store_true',
                                          help="Run model inference.")
        self.protocols_group.add_argument("--precompute_features", dest="precompute_features", required=False, default=False,
                                          action='store_true',
                                          help="Explicitly run feature pre-computation for cache-enabled datasets like interaction_graph_legacy.")
        self.protocols_group.add_argument("--dry_run", dest="dry_run", required=False, default=False,
                                          action='store_true',
                                          help="Check model setup but do not run inference.")

        # Common I/O options
        self.common_group = self.parser.add_argument_group('Common I/O')

        self.common_group.add_argument("--random_seed", dest="random_seed",
                                       required=False, default=41922, type=int,
                                       help="Specify a fixed random seed for reproducibility; if 'None', not reproducible without logging the random seed.")
        self.common_group.add_argument("--load_model", dest="load_model", required=False, default=None,
                                       help="Load a model for inference.")
        self.common_group.add_argument("--load_normalization_data_pkl", dest="load_normalization_data_pkl", required=False, default=None,
                                       help="The .pkl file from which normalization file from training will be loaded for "
                                            "inference.")
        self.common_group.add_argument("--skip_normalization",dest="skip_normalization", required=False, default=False,
                                       action="store_true", help="Skip normalization when loading datasets directly; useful if datasets "
                                                                 "have already been combined and normalized.")
        self.common_group.add_argument("--device", dest="device", required=False, default=None,
                                          help="Override GPU autosetup and specify device")
        self.common_group.add_argument("--debug", dest="debug", required=False, default=False,
                                          action='store_true', help="Extra print statements for help debug models.")
        self.common_group.add_argument("--log_filename_prefix", dest="log_filename_prefix", required=False, default=None,
                                          help="Prefix appended to logger filenames.")
        self.common_group.add_argument("--log_dir", dest="log_dir", required=False, default="./",
                                          help="Directory in which to save log files.")

        # Input files
        self.input_group = self.parser.add_argument_group('Input')
        self.input_group.add_argument("--input_ligand_file", dest="input_ligand_file", required=False,
                                       help="Input file containing newline-separated list of SDF and/or PDB files from which to "
                                            "generate a dataset for model training, validation, testing, and/or inference. "
                                            "Often provided alongside 'input_protein_file' to create an interaction graph "
                                            "dataset.")
        self.input_group.add_argument("--input_protein_file", dest="input_protein_file", required=False,
                                       help="Input file containing newline-separated list of SDF and/or PDB files from which to "
                                            "generate a dataset for model training, validation, testing, and/or inference. "
                                            "Often provided alongside 'input_ligand_file' to create an interaction graph "
                                            "dataset.")
        self.input_group.add_argument("--input_ligand_protein_pair_file", dest="input_ligand_protein_pair_file", required=False,
                                       help="Input file formatted as a 2-column file delimited with a semicolon containing on each row an SDF and/or PDB file in "
                                            "both columns from which to generate a dataset for model training, validation, testing, and/or inference. "
                                            "Typically provided to create an interaction graph dataset.")
        self.input_group.add_argument("--inference_sample_indices", dest="inference_sample_indices", required=False,
                                      type=int, nargs="+", help="A list of sample indices to process during inference.")

        # Dataset options
        self.dataset_group = self.parser.add_argument_group('Dataset')
        self.dataset_group.add_argument("--dataset_type", dest="dataset_type", required=False, default='interaction_graph_legacy',
                                        choices=['interaction_graph_legacy'],
                                        help="Generate a PyTorch dataset from this type of input data. "
                                             "Options are as follows: \n"
                                             "interaction_graph_legacy - an interaction graph featurized by molecular interactions. \n")
        self.dataset_group.add_argument("--num_result_classes", dest="num_result_classes", required=False,
                                        type=int, default=8,
                                        help="Number of model output classes (defaults match pretrained CORDIAL).")
        self.dataset_group.add_argument("--cache_dir", dest="cache_dir", required=False, default="./",
                                        type=str, help="Directory to cache pre-computed features and atom vocabularies on disk to save RAM and ensure consistency.")
        self.dataset_group.add_argument("--search_method", dest="search_method", required=False, default="cdist",
                                        type=str, choices=['cdist', 'kdtree', 'balltree'],
                                        help="Method used to identify interacting atoms.")
        self.dataset_group.add_argument("--distance_cutoff", dest="distance_cutoff", required=False, default=16.0,
                                        type=float, help="Distance threshold for interacting atoms.")
        self.dataset_group.add_argument("--step_size", dest="step_size", required=False, default=0.25,
                                        type=float, help="Distance bin size.")
        self.dataset_group.add_argument("--num_distance_bins", dest="num_distance_bins", required=False, default=64,
                                        type=int, help="Number of discrete distance bins.")
        self.dataset_group.add_argument("--reduce_interaction_graph", dest="reduce_interaction_graph", required=False, default=False,
                                        action="store_true", help="Reduce the size of the interaction graph by removing redundant edges.")
        self.dataset_group.add_argument("--filter_obstructions", dest="filter_obstructions", required=False, default=False,
                                        action="store_true", help="Remove obstructed interactions")
        self.dataset_group.add_argument("--obstruction_tolerance", dest="obstruction_tolerance", required=False, default=0.0,
                                        type=float,
                                        help="Reduce size of atom vdw radius by this amount when determining whether "
                                             "an atom is obstructing the path between two other potentially interacting atoms. ")
        self.dataset_group.add_argument("--binning_scheme", dest="binning_scheme", required=False, default="signed_directional",
                                        type=str, choices=["unsigned", "signed", "signed_directional"],
                                        help="Determines how sign pairs between interacting atoms are pooled. \n"
                                             "unsigned - pooled into common distance bins irrespective of sign. \n"
                                             "signed - pooled into common distance bins according to sign pairs -/-, +/+, and -/+. \n"
                                             "signed_directional - same as signed but differentiating -/+ and +/-.")
        self.dataset_group.add_argument("--exclude_feature_columns", dest="exclude_feature_columns", nargs="+",
                                        required=False, default=None, type=int, help="Exclude feature columns when loading dataset.")
        self.dataset_group.add_argument("--num_workers", dest="num_workers",
                                       required=False, default=0, type=int,
                                       help="Set the number of workers for the data loader.")
        self.dataset_group.add_argument("--num_feature_computation_workers", dest="num_feature_computation_workers",
                                       required=False, default=1, type=int,
                                       help="Set the number of workers for feature computation.")

        # Training and validation
        self.training_validation = self.parser.add_argument_group('Training and validation')
        self.training_validation.add_argument("--model_type", dest="model_type", required=False, default='cordial',
                                       help="Model type (defaults to 'cordial').",
                                              choices=['cordial'])

        # Inference
        self.inference = self.parser.add_argument_group('Inference')
        self.inference.add_argument("--parity", dest="parity", required=False, default=[0.5], nargs="+", type=float,
                              help="This is the model output value at or above which a classification task will return True (1) "
                                   "for that label; below the parity the returned value will be False (0).")

        # Common hyperparameters
        self.common_group.add_argument("--batch_method", "--bm", dest="batch_method", required=False,
                                                default='mini-batch',choices=['full-batch','mini-batch','online-learning'],
                                                help="Specify the batching style during training; if using mini-batch then also set the 'batch_size' "
                                                     "accordingly; note that mini-batch with a batch_size of 1 is equivalent to 'online-learning', and "
                                                     "'mini-batch' with a batch_size of the total number of training samples is equivalent to "
                                                     "full-batch. Consequently, this option is really just for convenience. Setting to full-batch or "
                                                     "online-learning will override the batch_size option.")
        self.common_group.add_argument("--batch_size", "--bs", dest="batch_size", required=False,
                                                default=32, help="Training batch size")
        
        # MLP-specific arguments
        self.mlp_group = self.parser.add_argument_group('MLP hyperparameters')
        self.mlp_group.add_argument("--hidden_layers", "--hl", dest="hidden_layers", nargs="+",
                                    required=False, type=int, default=[256, 256],
                                    help="Hidden layer sizes (defaults match pretrained CORDIAL: 256 256)")
        self.mlp_group.add_argument("--fc_dropout", "--d", dest="fc_dropout", nargs="+",
                                    required=False, type=float, default=[0.0, 0.25, 0.25],
                                    help="Dropout at input and hidden layers (defaults match pretrained CORDIAL)")
        self.mlp_group.add_argument("--activation_function_names", "--afn", dest="activation_function_names", nargs="+",
                                    required=False, type=str, default=['mish', 'mish'],
                                    help="Activation functions per hidden layer (defaults match pretrained CORDIAL)")
        self.mlp_group.add_argument("--output_activation_function_name", "--oafn", dest="output_activation_function_name",
                                    required=False, type=str, default=None,
                                    help="Activation function for output")

        # CORDIAL-specific arguments; currently handles InteractionGAT options
        self.cordial_group = self.parser.add_argument_group('CORDIAL hyperparameters')
        self.cordial_group.add_argument("--attention_type", "--at", dest="attention_type",
                                    required=False, default='axial', type=str, choices=['standard', 'axial'],
                                    help="Type of attention to use")
        self.cordial_group.add_argument("--attention_heads", "--ah", dest="attention_heads",
                                    required=False, default=1, type=int, help="Number of heads for standard attention")
        self.cordial_group.add_argument("--num_row_attn_heads", "--nrah", dest="num_row_attn_heads",
                                    required=False, default=2, type=int, help="Row attention heads (defaults match pretrained CORDIAL)")
        self.cordial_group.add_argument("--num_column_attn_heads", "--ncah", dest="num_column_attn_heads",
                                    required=False, default=1, type=int, help="Column attention heads (defaults match pretrained CORDIAL)")
        self.cordial_group.add_argument("--attention_dropout", "--ad", dest="attention_dropout",
                                    required=False, default=0.15, type=float,
                                    help="Attention dropout (defaults match pretrained CORDIAL)")
        self.cordial_group.add_argument("--num_attn_layers", "--nal", dest="num_attn_layers",
                                    required=False, default=2, type=int, help="Number of attention layers (defaults match pretrained CORDIAL)")
        self.cordial_group.add_argument("--transformer_embedding_expansion_factor", "--teef",
                                    dest="transformer_embedding_expansion_factor",
                                    required=False, default=4, type=int,
                                    help="Multiply the transformer embedding dimension by this factor following attention.")
        self.cordial_group.add_argument("--kernel_size", dest="kernel_size",
                                    required=False, default=7, type=int,
                                    help="The kernel size in the conv1d layer.")
        self.cordial_group.add_argument("--conv_dropout", dest="conv_dropout",
                                    required=False, default=0.05, type=float,
                                    help="Dropout applied after the conv1d layer.")
        self.cordial_group.add_argument("--conv_channels", dest="conv_channels", nargs="+",
                                    required=False, default=[4], type=int,
                                    help="Number of channels in the conv1d layer.")
        self.cordial_group.add_argument("--disable_conv1d", dest="disable_conv1d", required=False, default=False,
                                    action='store_true', help="Disable the convolutional layer")
        self.cordial_group.add_argument("--disable_attention", dest="disable_attention", required=False, default=False,
                                    action='store_true', help="Disable the attention layer")

    def str_to_bool(self, value):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            raise self.parser.ArgumentTypeError("Boolean value expected")

    def parse_args(self, args=None):
        return self.parser.parse_args(args)
