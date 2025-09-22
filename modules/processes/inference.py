#!/usr/bin/env python

# Python imports
import torch

# General imports
import numpy as np
from scipy.special import expit  # Sigmoid function
import csv

class Inference:
    """
    Class for running inference on a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model for inference.
        dataset_loader (torch.utils.data.DataLoader): DataLoader for the dataset to perform inference on.
        device (torch.device): Device on which the model will perform inference.
        output_basename (str): Base name for the output file. Defaults to "inference_results".
        model_type (str): The type of model being used. Defaults to 'mlp'.
        parity (list): List of threshold values for converting logits to binary labels for each class. Defaults to [0.5].
        binary_predict (bool): Perform a pure binary prediction based on parity values rather than a probabilistic prediction.
        num_result_classes (int): Number of classes for classification tasks. Defaults to 1.
        sample_indices (Optional[Union[int, List[int], range]]): Specific sample indices to process. 
            Can be a single index, list of indices, or range. Defaults to None (process all samples).

    Methods:
        run_inference(): Run inference on the provided dataset and return predictions.
        update_prediction_lists(predictions_list, predictions, predicted_class_logits_list, predicted_class_probs_list, predicted_class_indices_list): Update prediction lists.
        convert_logits_to_binary(logits): Convert logits to binary labels based on classification parity.
        convert_logits_to_probabilities(logits): Convert logits to probabilities using sigmoid.
        write_predictions_to_csv(predictions_list, predicted_class_indices_list, predicted_class_logits_list, predicted_class_probs_list): Write predictions and related information to a CSV file.
    """

    def __init__(self, model, dataset_loader, device, output_basename="inference_results", model_type='cordial',
                 parity=None, binary_predict=False, num_result_classes=1, sample_indices=None):
        """
        Initialize the Inference instance.

        Args:
            model (torch.nn.Module): The PyTorch model for inference.
            dataset_loader (torch.utils.data.DataLoader): DataLoader for the dataset to perform inference on.
            device (torch.device): Device on which the model will perform inference.
            output_basename (str): Base name for the output file. Defaults to "inference_results".
            model_type (str): The type of model being used. Defaults to 'mlp'.
            parity (list): List of threshold values for converting logits to binary labels for each class. Defaults to [0.5].
            binary_predict (bool): Perform a pure binary prediction based on parity values rather than a probabilistic prediction.
            num_result_classes (int): Number of classes for classification tasks. Defaults to 1.
            sample_indices (Optional[Union[int, List[int], range]]): Specific sample indices to process. 
                Can be a single index, list of indices, or range. Defaults to None (process all samples).
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.device = device
        self.output_basename = output_basename
        self.model_type = model_type
        self.parity = parity if parity is not None else [0.5]
        self.binary_predict = binary_predict
        self.num_result_classes = num_result_classes
        
        # Handle sample indices
        if isinstance(sample_indices, int):
            self.sample_indices = [sample_indices]
        else:
            self.sample_indices = sample_indices

    def run_inference(self):
        """
        Run inference on the provided dataset and return predictions.

        Returns:
            np.ndarray: Concatenated predictions.
            np.ndarray: Predicted class indices (if applicable).
        """
        self.model.eval()
        predictions_list = []
        predicted_class_logits_list = []
        predicted_class_probs_list = []
        predicted_class_indices_list = []
        original_indices_list = []

        with torch.no_grad():
            for batch_ii, batch in enumerate(self.dataset_loader):
                # Calculate which samples are in this batch
                batch_size = next(iter(batch.values())).size(0)
                batch_start_idx = batch_ii * batch_size
                batch_end_idx = batch_start_idx + batch_size

                # Skip this batch if we're only processing specific samples and none are in this batch
                if self.sample_indices is not None:
                    batch_indices = range(batch_start_idx, batch_end_idx)
                    if not any(idx in self.sample_indices for idx in batch_indices):
                        continue
                    
                    # Create mask for desired samples in this batch
                    # Create mask on CPU first
                    batch_mask = torch.tensor([i in self.sample_indices for i in batch_indices])
                    
                    # Apply mask to all batch tensors while they're still on CPU
                    batch = {k: v[batch_mask] if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    if len(batch[next(iter(batch))]) == 0:
                        continue

                # Move batch to device after masking
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}

                # Enable autocast only if CUDA is available
                autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.amp.autocast(autocast_device):
                    # Forward pass and obtain predictions based on model type
                    device_batch = {}
                    for k, v in batch.items():
                        if k == 'features' and isinstance(v, (list, tuple)):
                            # Handle tuple of tensors for embeddings
                            device_batch[k] = tuple(t.to(self.device, non_blocking=True) for t in v)
                        elif isinstance(v, torch.Tensor):
                            device_batch[k] = v.to(self.device, non_blocking=True)
                        else:
                            device_batch[k] = v # Keep non-tensors as-is
                    
                    predictions, _, _, _, _, _ = self.model(device_batch)
                    predictions_np = predictions.cpu().numpy()
                    
                    # Extract original indices if available
                    if 'original_index' in device_batch:
                        # original_index may be a tensor or list
                        oi = device_batch['original_index']
                        if isinstance(oi, torch.Tensor):
                            batch_original_indices = oi.cpu().numpy()
                        else:
                            batch_original_indices = oi
                        original_indices_list.extend(batch_original_indices)
                    else:
                        # Fallback to sequential indices if original indices not available
                        batch_size = predictions_np.shape[0]
                        batch_start_idx = len(predictions_list) if len(predictions_list) > 0 else 0
                        original_indices_list.extend(range(batch_start_idx, batch_start_idx + batch_size))

                    # Update prediction lists
                    (predictions_list, predicted_class_logits_list, predicted_class_probs_list, 
                     predicted_class_indices_list) = self.update_prediction_lists(
                        predictions_list, predictions_np, predicted_class_logits_list, 
                        predicted_class_probs_list, predicted_class_indices_list
                    )

        # Write predictions to CSV
        self.write_predictions_to_csv(predictions_list, predicted_class_indices_list, 
                                        predicted_class_logits_list, predicted_class_probs_list, 
                                        original_indices_list)

        predictions = np.concatenate(predictions_list, axis=0)
        predicted_class_indices = np.concatenate(predicted_class_indices_list, axis=0)

        return predictions, predicted_class_indices

    def update_prediction_lists(self, predictions_list, predictions, 
                                predicted_class_logits_list, predicted_class_probs_list, 
                                predicted_class_indices_list):
        """
        Convert predictions into their desired output format.

        Args:
            predictions_list (np.array): List of predictions to update.
            predictions (np.ndarray): Predicted values from the model.
            predicted_class_logits_list (list): Optional list to store predicted logits.
            predicted_class_probs_list (list): Optional list to store predicted probabilities.
            predicted_class_indices_list (list): Optional list to store predicted class indices.

        Returns:
            Updated predictions list,
            updated predicted class logits list (if applicable),
            updated predicted class probabilities list (if applicable),
            updated predicted class indices list (if applicable).
        """

        # CORDIAL was trained with BCEWithLogitsLoss for each result class
        predicted_class_probs = self.convert_logits_to_probabilities(predictions)
        predicted_class_indices = self.convert_logits_to_binary(predictions)
        predictions_list.append(predicted_class_probs)

        # Store additional prediction information
        if predicted_class_logits_list is not None:
            predicted_class_logits_list.append(predictions)
        if predicted_class_probs_list is not None:
            predicted_class_probs_list.append(predicted_class_probs)
        if predicted_class_indices_list is not None:
            predicted_class_indices_list.append(predicted_class_indices)

        return (predictions_list, predicted_class_logits_list, 
                predicted_class_probs_list, predicted_class_indices_list)

    def convert_logits_to_binary(self, logits):
        """
        Convert logits to binary labels based on classification parity.

        Args:
            logits (np.ndarray): Logits from the model.

        Returns:
            np.ndarray: Binary labels.
        """
        if len(self.parity) == 1:
            return (logits >= self.parity[0]).astype(int)
        else:
            if logits.ndim == 1:
                # Handle 1D input
                binary_labels = np.zeros_like(logits)
                for i, parity in enumerate(self.parity):
                    binary_labels[i] = (logits[i] >= parity).astype(int)
            else:
                # Handle 2D input
                binary_labels = np.zeros_like(logits)
                for i, parity in enumerate(self.parity):
                    binary_labels[:, i] = (logits[:, i] >= parity).astype(int)
            return binary_labels

    def convert_logits_to_probabilities(self, logits):
        """
        Convert logits to probabilities using sigmoid.

        Args:
            logits (np.ndarray): Logits from the model.

        Returns:
            np.ndarray: Probabilities.
        """
        return expit(logits)

    def write_predictions_to_csv(self, predictions_list, predicted_class_indices_list, 
                                 predicted_class_logits_list, predicted_class_probs_list,
                                 original_indices_list=None):
        """
        Write predictions and related information to a CSV file.

        Args:
            predictions_list (list): List of predicted values for each batch.
            predicted_class_indices_list (list): List of predicted class indices for each batch.
            predicted_class_logits_list (list): List of predicted class logits for each batch.
            predicted_class_probs_list (list): List of predicted class probabilities for each batch.
        """
        filename = f"{self.output_basename}_predictions.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write header
            header = ['Original_Index', 'Sequential_Index', 'Batch_Index', 'Batch_Item_Index', 'Predicted_Value', 'Predicted_Logits', 'Predicted_Probabilities', 'Predicted_Class_Index']
            csvwriter.writerow(header)
            
            # Write data
            sequential_index = 0
            for batch_idx, batch_predictions in enumerate(predictions_list):
                for i in range(len(batch_predictions)):
                    # Get original index if available
                    if original_indices_list is not None and sequential_index < len(original_indices_list):
                        original_idx = original_indices_list[sequential_index]
                    else:
                        original_idx = sequential_index  # Fallback
                    
                    # Handle scalar or array predictions
                    pred_value = batch_predictions[i]
                    if isinstance(pred_value, np.ndarray):
                        pred_str = ' '.join([f"{x:.4f}" for x in pred_value])
                    else:
                        pred_str = f"{pred_value:.4f}"
                    
                    # Format logits and probabilities as space-separated strings
                    logits_str, probs_str = "N/A", "N/A"
                    if predicted_class_logits_list is not None and predicted_class_logits_list[batch_idx] is not None:
                        logits = predicted_class_logits_list[batch_idx][i]
                        if isinstance(logits, np.ndarray):
                            logits_str = ' '.join([f"{x:.4f}" for x in logits])
                        else:
                            logits_str = f"{logits:.4f}"
                    
                    if predicted_class_probs_list is not None and predicted_class_probs_list[batch_idx] is not None:
                        probs = predicted_class_probs_list[batch_idx][i]
                        if isinstance(probs, np.ndarray):
                            probs_str = ' '.join([f"{x:.4f}" for x in probs])
                        else:
                            probs_str = f"{probs:.4f}"
                    
                    row = [original_idx, sequential_index, batch_idx, i, pred_str, f"[{logits_str}]", f"[{probs_str}]", predicted_class_indices_list[batch_idx][i]]
                    csvwriter.writerow(row)
                    sequential_index += 1
        
        print(f"Predictions written to {filename}")

