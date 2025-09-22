#!/usr/bin/env python

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.linalg import solve
from scipy.spatial.distance import cdist

# Project imports
from modules.features.atom_data import pauling_electronegativity, polarizabilities, hardness, hybridization

"""
atom_properties_from_molecule.py

This script defines convenience functions that each return a list of properties for each atom in a molecule.
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
def atomic_number(molecule):
    """
    Compute the atomic numbers for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of atomic numbers.
    """
    sanity_check(molecule)
    atomic_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    return atomic_numbers

# Atomic mass
def atomic_mass(molecule):
    """
    Compute the atomic masses for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of atomic masses.
    """
    sanity_check(molecule)
    atomic_masses = [atom.GetMass() for atom in molecule.GetAtoms()]
    return atomic_masses

# Hybridization state
def hybridization_state(molecule):
    """
    Compute the hybridization states for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of hybridization states.
    """
    sanity_check(molecule)
    hybridizations = [hybridization.get(atom.GetHybridization()) for atom in molecule.GetAtoms()]
    return hybridizations

# Formal charge
def formal_charge(molecule):
    """
    Compute the formal charges for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of formal charges.
    """
    sanity_check(molecule)
    formal_charges = [atom.GetFormalCharge() for atom in molecule.GetAtoms()]
    return formal_charges

# Number of hydrogens
def number_hydrogens(molecule):
    """
    Compute the number of hydrogens for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of the number of hydrogens.
    """
    sanity_check(molecule)
    num_hydrogens = [atom.GetTotalNumHs() for atom in molecule.GetAtoms()]
    return num_hydrogens

# Chirality
def chirality(molecule):
    """
    Compute the chirality for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of chirality values.
    """
    sanity_check(molecule)
    chiralities = [atom.GetChiralTag() for atom in molecule.GetAtoms()]
    return chiralities

# Van der Waals radius
def vdw_radius(molecule):
    """
    Compute the Van der Waals radii for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of Van der Waals radii.
    """
    sanity_check(molecule)
    vdw_radii = [Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) for atom in molecule.GetAtoms()]
    return vdw_radii

def polarized_vdw_radius(molecule):
    """
    Compute the Van der Waals radii for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of Van der Waals radii.
    """
    sanity_check(molecule)
    polarized_vdw_radii = [Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) * polarizabilities.get(atom.GetAtomicNum(),0) for atom in molecule.GetAtoms()]
    return polarized_vdw_radii

# Electronegativity
def electronegativity(molecule):
    """
    Compute the electronegativities for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of electronegativities.
    """
    sanity_check(molecule)
    electronegativities = [pauling_electronegativity.get(atom.GetAtomicNum()) for atom in molecule.GetAtoms()]
    electronegativities = [-1 if e is None else e for e in electronegativities]
    return electronegativities

def eneg_difference_carbon(molecule):
    """
    Compute the electronegativities for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of electronegativities.
    """
    sanity_check(molecule)
    c_eneg = pauling_electronegativity.get(6)
    electronegativities = [pauling_electronegativity.get(atom.GetAtomicNum()) - c_eneg for atom in molecule.GetAtoms()]
    electronegativities = [-1 if e is None else e for e in electronegativities]
    return electronegativities

# Aromaticity
def aromaticity(molecule):
    """
    Compute the aromaticity for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of aromaticity values.
    """
    sanity_check(molecule)
    aromaticities = [atom.GetIsAromatic() for atom in molecule.GetAtoms()]
    return aromaticities

def is_aromatic_ternary(molecule):
    """
    Compute the aromaticity ternary for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List 1 for aromatic atoms, -1 for non-aromatic atoms.
    """
    sanity_check(molecule)
    is_aromatic_ternary_result = [1 if atom.GetIsAromatic() else -1 for atom in molecule.GetAtoms()]
    return is_aromatic_ternary_result

# Degree
def degree(molecule):
    """
    Compute the degree for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of degrees.
    """
    sanity_check(molecule)
    degrees = [atom.GetDegree() for atom in molecule.GetAtoms()]
    return degrees

# Valence
def valence(molecule):
    """
    Compute the valence for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of valences.
    """
    sanity_check(molecule)
    valences = [atom.GetTotalValence() for atom in molecule.GetAtoms()]
    return valences

# Partial charge
def gasteiger_charges(molecule, recompute=True):
    """
    Compute the Gasteiger charges for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.
    - recompute (bool): Whether to recompute the charges.

    Returns:
    - list: List of Gasteiger charges.
    """
    sanity_check(molecule)
    if recompute:
        AllChem.ComputeGasteigerCharges(molecule)
    partial_charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in molecule.GetAtoms()]
    return partial_charges

def polarized_gasteiger_charges(molecule, recompute=True):
    """
    Compute the Gasteiger charges weighted by atomic polarizability for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.
    - recompute (bool): Whether to recompute the charges.

    Returns:
    - list: List of Gasteiger charges.
    """
    sanity_check(molecule)
    if recompute:
        AllChem.ComputeGasteigerCharges(molecule)
    polarized_partial_charges = [atom.GetDoubleProp('_GasteigerCharge') * polarizabilities.get(atom.GetAtomicNum(),0) for atom in molecule.GetAtoms()]
    return polarized_partial_charges

def eem_charges(molecule):
    """
    Compute the EEM charges for each atom in the molecule.

    This method uses the Electronegativity Equalization Method (EEM) to calculate partial charges.
    EEM is based on the principle that atoms in a molecule adjust their electron densities
    to equalize their electronegativity.

    The EEM is mathematically represented as:
        χ_i + 0.5 * η_i * q_i + Σ_j (q_j / R_ij) = λ
    where:
        - χ_i is the electronegativity of atom i
        - η_i is the hardness of atom i
        - q_i is the partial charge on atom i
        - R_ij is the distance between atoms i and j
        - λ is a Lagrange multiplier ensuring charge conservation

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: List of EEM charges.

    References:
    - Mortier, W. J., Ghosh, S. K., & Shankar, S. (1986). Electronegativity Equalization Method for the Calculation of Atomic Charges in Molecules. Journal of the American Chemical Society, 108(15), 4315-4320.
    - Pearson, R. G. (1988). Absolute electronegativity and hardness: application to inorganic chemistry. Inorganic Chemistry, 27(4), 734-740.
    """
    sanity_check(molecule)
    num_atoms = molecule.GetNumAtoms()
    A = np.zeros((num_atoms, num_atoms))
    b = np.zeros(num_atoms)
    atom_indices = range(num_atoms)

    conf = molecule.GetConformer()

    positions = np.array([conf.GetAtomPosition(i) for i in atom_indices])
    distances = cdist(positions, positions)
    np.fill_diagonal(distances, np.inf)

    for i in atom_indices:
        atom_i = molecule.GetAtomWithIdx(i)
        atomic_num_i = atom_i.GetAtomicNum()
        chi_i = pauling_electronegativity.get(atomic_num_i)
        eta_i = hardness.get(atomic_num_i)
        b[i] = -chi_i
        A[i, i] = eta_i / 2.0

    inv_distances = 1.0 / distances
    np.fill_diagonal(inv_distances, 0)
    A += inv_distances

    charges = solve(A, b)
    return charges

def polarizability(molecule):
    """
    Compute the atomic polarizability for each atom in the molecule based on empirical atomic polarizabilities.
    Polarizabilities are sourced from Choudhary et al., 2019, for the first 103 elements.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - dict: Dictionary with atom indices as keys and polarizability values as values.
    """
    sanity_check(molecule)
    atom_polarizabilities = [polarizabilities.get(atom.GetAtomicNum(),0) for atom in molecule.GetAtoms()]  # Default 0 for unlisted elements
    return atom_polarizabilities

# Determine if an atom is a hydrogen atom
def is_h_ternary(molecule):
    """
    Determine if each atom in a molecule is a hydrogen atom.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: A list of integers where 1 indicates an atom is a hydrogen atom, and -1 indicates it is not.
    """
    sanity_check(molecule)
    is_h_ternary_result = [1 if atom.GetAtomicNum() == 1 else -1 for atom in molecule.GetAtoms()]
    return is_h_ternary_result

# Determine if an atom is a hydrogen bond donor or acceptor
def is_h_bond_donor_ternary(molecule):
    """
    Determine if each atom in a molecule is a hydrogen bond donor or acceptor.

    A hydrogen bond donor is identified as a nitrogen or oxygen atom bonded to at least one hydrogen.
    A hydrogen bond acceptor is identified as a nitrogen or oxygen not bonded to a hydrogen (simplified model).

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: A list of integers where 1 indicates a hydrogen bond donor, -1 indicates a hydrogen bond acceptor,
            and 0 indicates neither.
    """
    sanity_check(molecule)
    is_h_bond_donor_ternary_result = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() in [7, 8] and any(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()):
            is_h_bond_donor_ternary_result.append(1)  # Hydrogen bond donor
        elif atom.GetAtomicNum() in [7, 8]:
            is_h_bond_donor_ternary_result.append(-1)  # Hydrogen bond acceptor
        else:
            is_h_bond_donor_ternary_result.append(0)  # Otherwise
    return is_h_bond_donor_ternary_result

# Determine if an atom is hydrophobic
def is_hydrophobic_ternary(molecule):
    """
    Determine if each atom in a molecule is hydrophobic.

    In this model:
    - Carbon atoms not bonded to nitrogen, oxygen, or single fluorine atoms are considered hydrophobic.
    - Sulfur and phosphorus are considered hydrophobic if not in a highly polar environment.
    - Hydrogen atoms are considered neutral.
    - Nitrogen, oxygen, and single fluorine atoms are considered hydrophilic.
    - Carbon and fluorine atoms in -CFX groups are considered hydrophobic, where X can be any combination of halogen.

    This function returns 1 for hydrophobic atoms, -1 for non-hydrophobic atoms, and 0 for hydrogen atoms or heteroatoms not explicitly considered here.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: A list of integers indicating hydrophobicity: 1 for hydrophobic, -1 for not hydrophobic, and 0 for hydrogen.
    """

    sanity_check(molecule)
    is_hydrophobic_ternary_result = []
    for atom in molecule.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 1:  # Hydrogen (H)
            is_hydrophobic_ternary_result.append(0)
        elif atomic_num == 6:  # Carbon (C)
            # Check if carbon is part of a -CFX group
            if len([neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() in [9,17,35,53]]) == 3:
                is_hydrophobic_ternary_result.append(1)  # CF3 group
            elif all(neighbor.GetAtomicNum() not in [7, 8, 9] for neighbor in atom.GetNeighbors()):
                is_hydrophobic_ternary_result.append(1)
            else:
                is_hydrophobic_ternary_result.append(-1)
        elif atomic_num in [15,16]:  # Sulfur (S), Phosphorous (P)
            if all(neighbor.GetAtomicNum() not in [7, 8, 9] for neighbor in atom.GetNeighbors()):
                is_hydrophobic_ternary_result.append(1)
            else:
                is_hydrophobic_ternary_result.append(-1)
        elif atomic_num == 9:  # Fluorine (F)
            # Check if fluorine is part of a -CFX group
            if any(neighbor.GetAtomicNum() == 6 and len([n for n in neighbor.GetNeighbors() if n.GetAtomicNum() in [9,17,35,53]]) == 3 for neighbor in
                   atom.GetNeighbors()):
                is_hydrophobic_ternary_result.append(1)
            else:
                is_hydrophobic_ternary_result.append(-1)
        elif atomic_num in [7,8]:  # Nitrogen (N) or Oxygen (O)
            is_hydrophobic_ternary_result.append(-1)
        else:
            is_hydrophobic_ternary_result.append(0)
    return is_hydrophobic_ternary_result

def polarized_hydrophobic_ternary(molecule):
    """
    Compute the Gasteiger charges for each atom in the molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.
    - recompute (bool): Whether to recompute the charges.

    Returns:
    - list: List of Gasteiger charges.
    """
    sanity_check(molecule)
    is_hydrophobic_ternary_result = is_hydrophobic_ternary(molecule)
    polarized_partial_charges = [is_hydrophobic_ternary_result[a_i] * polarizabilities.get(atom.GetAtomicNum(),0) for a_i, atom in enumerate(
        molecule.GetAtoms())]
    return polarized_partial_charges

def is_in_ring_ternary(molecule):
    """
    Determine if each atom in a molecule is part of a ring structure.

    This function identifies heavy atoms (non-hydrogen) that are part of ring structures within the molecule.
    It returns 1 for heavy atoms that are in a ring, -1 for heavy atoms not in a ring, and 0 for hydrogen atoms.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: A list of integers where 1 indicates a heavy atom is part of a ring, -1 indicates a heavy atom is not
            part of a ring, and 0 indicates the atom is a hydrogen atom.
    """
    sanity_check(molecule)
    is_in_ring_ternary_result = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Hydrogen check
            is_in_ring_ternary_result.append(0)
        else:  # Heavy atom check
            is_in_ring_ternary_result.append(1 if atom.IsInRing() else -1)
    return is_in_ring_ternary_result

def is_in_aromatic_ring_ternary(molecule):
    """
    Determine if each heavy atom in a molecule is part of an aromatic ring.

    This function returns:
    - 1 if the heavy atom is part of an aromatic ring,
    - -1 if the heavy atom is part of a non-aromatic ring or not in any ring,
    - 0 if the atom is a hydrogen atom.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.

    Returns:
    - list: A list of integers where 1 indicates a heavy atom is part of an aromatic ring, -1 indicates a heavy atom
            is not part of an aromatic ring, and 0 indicates the atom is a hydrogen atom.
    """
    sanity_check(molecule)
    results = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Hydrogen check
            results.append(0)
        elif atom.IsInRing():
            if atom.GetIsAromatic():
                results.append(1)  # Atom is in an aromatic ring
            else:
                results.append(-1)  # Atom is in a ring, but it is not aromatic
        else:
            results.append(-1)  # Atom is not in any ring
    return results


def compute_atomic_properties(molecule, property_names):
    """
    Compute atomic properties for a given molecule.

    Parameters:
    - molecule (Chem.Mol): RDKit molecule object.
    - property_names (list): List of strings specifying the desired properties.

    Returns:
    - dict: Dictionary with property names as keys and computed values as values.
    """
    result_dict = {}

    for prop_name in property_names:
        if prop_name == 'atomic_number':
            result_dict[prop_name] = atomic_number(molecule)
        elif prop_name == 'atomic_mass':
            result_dict[prop_name] = atomic_mass(molecule)
        elif prop_name == 'hybridization_state':
            result_dict[prop_name] = hybridization_state(molecule)
        elif prop_name == 'formal_charge':
            result_dict[prop_name] = formal_charge(molecule)
        elif prop_name == 'number_hydrogens':
            result_dict[prop_name] = number_hydrogens(molecule)
        elif prop_name == 'chirality':
            result_dict[prop_name] = chirality(molecule)
        elif prop_name == 'polarized_vdw_radius':
            result_dict[prop_name] = polarized_vdw_radius(molecule)
        elif prop_name == 'vdw_radius':
            result_dict[prop_name] = vdw_radius(molecule)
        elif prop_name == 'electronegativity':
            result_dict[prop_name] = electronegativity(molecule)
        elif prop_name == 'eneg_difference_carbon':
            result_dict[prop_name] = eneg_difference_carbon(molecule)
        elif prop_name == 'aromaticity':
            result_dict[prop_name] = aromaticity(molecule)
        elif prop_name == 'is_aromatic_ternary':
            result_dict[prop_name] = is_aromatic_ternary(molecule)
        elif prop_name == 'degree':
            result_dict[prop_name] = degree(molecule)
        elif prop_name == 'valence':
            result_dict[prop_name] = valence(molecule)
        elif prop_name == 'polarized_gasteiger_charges':
            result_dict[prop_name] = polarized_gasteiger_charges(molecule)
        elif prop_name == 'gasteiger_charges':
            result_dict[prop_name] = gasteiger_charges(molecule)
        elif prop_name == 'eem_charges':
            result_dict[prop_name] = eem_charges(molecule)
        elif prop_name == 'polarizability':
            result_dict[prop_name] = polarizability(molecule)
        elif prop_name == 'is_h_ternary':
            result_dict[prop_name] = is_h_ternary(molecule)
        elif prop_name == 'is_h_bond_donor_ternary':
            result_dict[prop_name] = is_h_bond_donor_ternary(molecule)
        elif prop_name == 'polarized_hydrophobic_ternary':
            result_dict[prop_name] = polarized_hydrophobic_ternary(molecule)
        elif prop_name == 'is_hydrophobic_ternary':
            result_dict[prop_name] = is_hydrophobic_ternary(molecule)
        elif prop_name == 'is_in_ring_ternary':
            result_dict[prop_name] = is_in_ring_ternary(molecule)
        elif prop_name == 'is_in_aromatic_ring_ternary':
            result_dict[prop_name] = is_in_aromatic_ring_ternary(molecule)
        else:
            print(f"Warning: Unsupported property '{prop_name}'")

    return result_dict
