#!/usr/bin/env python

import numpy as np
from rdkit import Chem
from modules.features.atom_data import pauling_electronegativity, polarizabilities, hardness

from modules.features.atom_properties_from_molecule import (
    atomic_number,
    atomic_mass,
    hybridization_state,
    formal_charge,
    number_hydrogens,
    chirality,
    vdw_radius,
    polarized_vdw_radius,
    electronegativity,
    eneg_difference_carbon,
    aromaticity,
    degree,
    valence,
    gasteiger_charges,
    polarized_gasteiger_charges,
    eem_charges,
    polarizability,
    is_h_ternary,
    is_h_bond_donor_ternary,
    is_hydrophobic_ternary,
    polarized_hydrophobic_ternary,
    is_in_ring_ternary,
    is_in_aromatic_ring_ternary
)

from modules.features.atom_properties_from_molecules import (
    eem_charges_with_external
)

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

def unified_compute_atomic_properties(molecule, property_names, external_molecules=None, distance_cutoff=16.0):
    """
    Compute atomic properties for a given molecule, considering external molecules if required.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.
    - property_names (list): List of strings specifying the desired properties.
    - external_molecules (list[Chem.Mol], optional): List of external RDKit molecule objects. Required for some properties.
    - distance_cutoff (float, optional): Distance cutoff for considering contributions from external molecules (default is 16.0 Å).

    Returns:
    - dict: Dictionary with property names as keys and computed values as values.
    """
    result_dict = {}

    for prop_name in property_names:
        if prop_name == 'eem_charges_with_external':
            if external_molecules is None:
                raise ValueError(f"External molecules are required for property '{prop_name}'")
            result_dict[prop_name] = eem_charges_with_external(molecule, external_molecules, distance_cutoff)
        elif prop_name in ['atomic_number', 'atomic_mass', 'hybridization_state', 'formal_charge', 'number_hydrogens', 
                           'chirality', 'vdw_radius', 'polarized_vdw_radius', 'electronegativity', 'eneg_difference_carbon', 
                           'aromaticity', 'degree', 'valence', 'gasteiger_charges', 'polarized_gasteiger_charges', 
                           'eem_charges', 'polarizability', 'is_h_ternary', 'is_h_bond_donor_ternary', 'is_hydrophobic_ternary', 
                           'polarized_hydrophobic_ternary', 'is_in_ring_ternary', 'is_in_aromatic_ring_ternary']:
            result_dict[prop_name] = globals()[prop_name](molecule)
        else:
            print(f"Warning: Unsupported property '{prop_name}'")

    return result_dict

