from typing import List, Union
from enum import Enum
import numpy as np

# Enum Class for Binning Strategies
class BinStrategy(Enum):
    """Enumeration of available binning strategies."""
    INTERVAL = "interval"  # Current strategy: [a,b)
    NEAREST = "nearest"    # Assign to nearest bin center
    THRESHOLD = "threshold"  # Values below min_value get all zeros, above max_value get last bin=1

# Enum Class for Fuzzy Binning Strategies
class FuzzyBinStrategy(Enum):
    """Enumeration of available fuzzy binning strategies."""
    PROPORTIONAL = "proportional"  # Split between two nearest bins
    GAUSSIAN = "gaussian"          # Gaussian distribution around value

def generate_bins(min_value: float, max_value: float, discretization: float) -> List[float]:
    """
    Generates a list of bin edges based on the provided minimum value, maximum value, and discretization step.

    Parameters:
    ----------
    min_value : float
        The minimum value of the range.
    max_value : float
        The maximum value of the range.
    discretization : float
        The step size between each bin edge.

    Returns:
    -------
    List[float]
        A list of bin edges.
    """
    return [min_value + i * discretization for i in range(int((max_value - min_value) / discretization) + 1)]

def bin_value(value: float, bins: List[float], strategy: Union[str, BinStrategy] = BinStrategy.INTERVAL) -> List[int]:
    """
    Bins a given value into one of the specified bins, producing a one-hot encoded vector.

    Parameters:
    ----------
    value : float
        The value to bin.
    bins : List[float]
        The list of bin edges.
    strategy : Union[str, BinStrategy]
        The binning strategy to use. Options:
        - 'interval': Traditional [a,b) binning
        - 'nearest': Assign to nearest bin center
        - 'threshold': Values below min_value get all zeros, above max_value get last bin=1

    Returns:
    -------
    List[int]
        A one-hot encoded vector representing the binned category of the value.
    """
    # Create one-hot vector with length equal to number of bins (not bin edges)
    one_hot = [0] * (len(bins) - 1)
    
    if isinstance(strategy, str):
        strategy = BinStrategy(strategy)
    
    if strategy == BinStrategy.NEAREST:
        # For nearest strategy, we compare to bin centers
        bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        # Find nearest center
        nearest_idx = min(range(len(bin_centers)), 
                         key=lambda i: abs(bin_centers[i] - value))
        one_hot[nearest_idx] = 1
    
    elif strategy == BinStrategy.THRESHOLD:
        # For threshold strategy:
        # - If value < min_value, all bins are 0
        # - If value >= max_value, last bin is 1
        # - Otherwise, use standard interval binning
        if value < bins[0]:
            # All bins remain 0
            pass
        elif value >= bins[-1]:
            one_hot[-1] = 1
        else:
            for i in range(len(bins) - 1):
                if bins[i] <= value < bins[i+1]:
                    one_hot[i] = 1
                    break
        
    else:  # BinStrategy.INTERVAL
        if value < bins[0]:
            one_hot[0] = 1
        elif value >= bins[-1]:
            one_hot[-1] = 1
        else:
            for i in range(len(bins) - 1):
                if bins[i] <= value < bins[i+1]:
                    one_hot[i] = 1
                    break
                    
    return one_hot

def fuzzy_bin_value(value: float, bins: List[float], strategy: Union[str, FuzzyBinStrategy] = FuzzyBinStrategy.PROPORTIONAL, 
                   sigma: float = 1.0) -> List[float]:
    """
    Bins a given value using fuzzy binning strategies, producing a vector of bin weights.

    Parameters:
    ----------
    value : float
        The value to bin.
    bins : List[float]
        The list of bin edges.
    strategy : Union[str, FuzzyBinStrategy]
        The fuzzy binning strategy to use. Options:
        - 'prop': Split between two nearest bins proportionally
        - 'fuzzy': Gaussian distribution around value
    sigma : float
        Standard deviation for Gaussian distribution (only used with FUZZY strategy)

    Returns:
    -------
    List[float]
        A vector of weights for each bin, summing to 1.0
    """
    bin_vector = [0.0] * len(bins)
    
    if isinstance(strategy, str):
        strategy = FuzzyBinStrategy(strategy)
    
    if strategy == FuzzyBinStrategy.PROPORTIONAL:
        if value <= bins[0]:
            bin_vector[0] = 1.0
        elif value >= bins[-1]:
            bin_vector[-1] = 1.0
        else:
            # Find the two nearest bins
            for i in range(len(bins) - 1):
                if bins[i] <= value < bins[i+1]:
                    # Calculate proportional distance between bins
                    distance = bins[i+1] - bins[i]
                    weight = (bins[i+1] - value) / distance
                    bin_vector[i] = weight
                    bin_vector[i+1] = 1.0 - weight
                    break
    
    elif strategy == FuzzyBinStrategy.GAUSSIAN:
        # Calculate bin centers
        bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        bin_centers = [bins[0]] + bin_centers + [bins[-1]]
        
        # Calculate Gaussian weights
        weights = np.exp(-0.5 * ((np.array(bin_centers) - value) / sigma) ** 2)
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        bin_vector = weights.tolist()
    
    return bin_vector

def convert_values_to_bins(values: List[Union[float, str]], min_value: float, max_value: float, 
                                   discretization: float, strategy: Union[str, BinStrategy, FuzzyBinStrategy] = BinStrategy.INTERVAL,
                                   sigma: float = 1.0) -> List[List[float]]:
    """
    Process a list of values into binned vectors using either hard or fuzzy binning.

    Parameters:
    ----------
    values : List[Union[float, str]]
        List of values to process. Can contain strings that can be converted to floats.
    min_value : float
        The minimum value of the binning range.
    max_value : float
        The maximum value of the binning range.
    discretization : float
        The discretization step for binning.
    strategy : Union[str, BinStrategy, FuzzyBinStrategy]
        The binning strategy to use.
    sigma : float
        Standard deviation for Gaussian distribution (only used with FUZZY strategy)

    Returns:
    -------
    List[List[float]]
        List of binned vectors for each input value.
    """
    bins = generate_bins(min_value, max_value, discretization)
    result = []
    
    for val in values:
        try:
            value = float(val)
            if isinstance(strategy, (str, FuzzyBinStrategy)) and strategy in [FuzzyBinStrategy.PROPORTIONAL, FuzzyBinStrategy.GAUSSIAN, "proportional", "gaussian"]:
                binned_vector = fuzzy_bin_value(value, bins, strategy, sigma)
            else:
                binned_vector = bin_value(value, bins, strategy)
            result.append(binned_vector)
        except (ValueError, TypeError):
            result.append([0.0] * len(bins))  # Handle invalid values with zero vector
            
    return result
