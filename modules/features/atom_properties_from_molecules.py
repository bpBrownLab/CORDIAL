#!/usr/bin/env python

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.linalg import solve
from scipy.spatial.distance import cdist

# Project imports
from modules.features.atom_data import pauling_electronegativity, polarizabilities, hardness

"""
atom_properties_from_molecules.py

This script defines functions that each return a list of properties for each atom in a molecule. 
The properties that are computed are influenced by / receive contributions from external molecules.  
"""

def sanity_check(molecule):
    """
    Perform sanity checks on the input molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Raises:
    - ValueError: If the molecule is invalid or of an unsupported type.
    """
    if molecule is None:
        raise ValueError("Invalid molecule")
    elif not isinstance(molecule, Chem.Mol):
        raise ValueError("Invalid molecule type. Supported types: RDKit Mol")

# Atomic number
def eem_charges_with_external(target_molecule, external_molecules, distance_cutoff=16.0):
    """
    Compute the EEM charges for each atom in the target molecule, considering contributions from external molecules.

    This method uses the Electronegativity Equalization Method (EEM) to calculate partial charges.
    EEM is based on the principle that atoms in a molecule adjust their electron densities
    to equalize their electronegativity. This variant includes contributions from external molecules.

    The EEM is mathematically represented as:
        χ_i + 0.5 * η_i * q_i + Σ_j (q_j / R_ij) = λ
    where:
        - χ_i is the electronegativity of atom i
        - η_i is the hardness of atom i
        - q_i is the partial charge on atom i
        - R_ij is the distance between atoms i and j
        - λ is a Lagrange multiplier ensuring charge conservation

    Parameters:
    - target_molecule (Chem.Mol): RDKit molecule object for the target molecule.
    - external_molecules (list of Chem.Mol): List of RDKit molecule objects for the external molecules.
    - distance_cutoff (float): Distance cutoff for considering contributions from external molecules (default is 16.0 Å).

    Returns:
    - list: List of EEM charges for the target molecule.

    References:
    - Mortier, W. J., Ghosh, S. K., & Shankar, S. (1986). Electronegativity Equalization Method for the Calculation of Atomic Charges in Molecules. Journal of the American Chemical Society, 108(15), 4315-4320.
    - Pearson, R. G. (1988). Absolute electronegativity and hardness: application to inorganic chemistry. Inorganic Chemistry, 27(4), 734-740.
    """

    sanity_check(target_molecule)
    for ext_mol in external_molecules:
        sanity_check(ext_mol)

    num_atoms_target = target_molecule.GetNumAtoms()
    num_atoms_total = num_atoms_target + sum([mol.GetNumAtoms() for mol in external_molecules])

    A = np.zeros((num_atoms_total, num_atoms_total))
    b = np.zeros(num_atoms_target)
    atom_indices_target = range(num_atoms_target)

    conf_target = target_molecule.GetConformer()
    confs_external = [mol.GetConformer() for mol in external_molecules]

    # Get positions of all atoms in the target molecule
    positions_target = np.array([conf_target.GetAtomPosition(i) for i in atom_indices_target])

    # Get positions of all atoms in the external molecules
    positions_external = np.concatenate(
        [np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]) for conf, mol in zip(confs_external, external_molecules)])

    # Combine positions
    positions_total = np.concatenate([positions_target, positions_external])

    # Calculate pairwise distances using broadcasting
    distances = cdist(positions_total, positions_total)

    # Avoid division by zero by setting diagonal to infinity
    np.fill_diagonal(distances, np.inf)

    # Apply distance cutoff
    distances[distances > distance_cutoff] = np.inf

    # Populate vector b and diagonal of matrix A for the target molecule
    for i in atom_indices_target:
        atom_i = target_molecule.GetAtomWithIdx(i)
        atomic_num_i = atom_i.GetAtomicNum()
        chi_i = pauling_electronegativity.get(atomic_num_i)
        eta_i = hardness.get(atomic_num_i)
        b[i] = -chi_i
        A[i, i] = eta_i / 2.0

    # Populate off-diagonal of matrix A for all atom interactions
    inv_distances = 1.0 / distances
    np.fill_diagonal(inv_distances, 0)  # Ensure the diagonal is zero
    A += inv_distances

    # Solve the linear system for the target molecule's atoms
    charges = solve(A[:num_atoms_target, :num_atoms_target], b)
    return charges

def compute_atomic_properties(molecule, external_molecules, distance_cutoff, property_names):
    """
    Compute atomic properties for a given molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.
    - external_molecules (list[Chem.Mol]): list of RDKit molecule objects.
    - distance_cutoff (float): distance cutoff value for external molecule atoms.
    - property_names (list): List of strings specifying the desired properties.

    Returns:
    - dict: Dictionary with property names as keys and computed values as values.
    """
    result_dict = {}

    for prop_name in property_names:
        if prop_name == 'eem_charges_with_external':
            result_dict[prop_name] = eem_charges_with_external(molecule, external_molecules, distance_cutoff)
        else:
            print(f"Warning: Unsupported property '{prop_name}'")

    return result_dict
