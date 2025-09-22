#!/usr/bin/env python

"""
interaction_graph_builder.py

This script defines a class, InteractionGraphBuilder, that builds an interaction graph between two molecules based on a user-specified distance cutoff.
The class allows the user to choose between three distance calculation methods: 'cdist' (using SciPy), 'kdtree' (using SciPy's cKDTree), or 'balltree' (using scikit-learn's BallTree).
"""

import numpy as np
from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
from Bio.PDB.Structure import Structure
from Bio.PDB import Selection
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import torch
from typing import List, Tuple, Optional

class InteractionGraphBuilder:
    """Build interaction graphs between molecular pairs."""
    
    def __init__(self, molecule1, molecule2, search_method='cdist', distance_cutoff=16.0, 
                 filter_obstructions=False, obstruction_tolerance=0.0, reduce_interaction_graph=True, 
                 debug=False, device='cuda'):
        """
        Initialize InteractionGraphBuilder.

        Parameters:
        - molecule1 (RDKit Mol or BioPython Structure): The first molecule.
        - molecule2 (RDKit Mol or BioPython Structure): The second molecule.
        - search_method (str): The method with which neighbor atoms will be identified ('cdist', 'kdtree', 'balltree').
        - distance_cutoff (float): The distance cutoff for interaction (default is 16.0 Å).
        - filter_obstruction (bool): If True, filter out interactions where another atom lies within the line segment of the interaction.
        - obstruction_tolerance (float): Subtract this value from the VDW radius of candidate obstructing atoms when checking for obstruction.
        - reduce_interaction_graph (bool): If True, reduce the interaction graph to include only interacting atoms.
        - debug (bool): Enable debug printing.
        - device (str): Device to use for GPU operations ('cuda' or 'cpu').
        """
        self.molecule1 = molecule1
        self.molecule2 = molecule2
        self.search_method = search_method
        self.distance_cutoff = distance_cutoff
        self.filter_obstructions = filter_obstructions
        self.obstruction_tolerance = obstruction_tolerance
        self.debug = debug
        
        # Set device for GPU operations
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        if search_method not in ['cdist', 'kdtree', 'balltree']:
            raise ValueError("Invalid search_method! Choose from 'cdist', 'kdtree', or 'balltree'.")

        # Build the full interaction graph
        full_interaction_graph, atom_pair_distances = self.build_interaction_graph()
        if self.debug:
            print(f"Full interaction graph size: {full_interaction_graph.shape}")
            print(f"Full atom pair distances size: {atom_pair_distances.shape}")

        # Reduce the interaction graph to include only interacting atoms
        if reduce_interaction_graph:
            self.interaction_graph, self.interacting_atoms1, self.interacting_atoms2, self.atom_pair_distances = self.reduce_interaction_graph(full_interaction_graph, atom_pair_distances)
            if self.debug:
                print(f"Reduced interaction graph size: {self.interaction_graph.shape}")
                print(f"Reduced atom pair distances size: {self.atom_pair_distances.shape}")
        else:
            # Convert to torch tensors for consistency
            self.interaction_graph = torch.tensor(full_interaction_graph, dtype=torch.float32)
            self.atom_pair_distances = torch.tensor(atom_pair_distances, dtype=torch.float32)
            self.interacting_atoms1 = np.arange(self.interaction_graph.shape[0]).tolist()
            self.interacting_atoms2 = np.arange(self.interaction_graph.shape[1]).tolist()

        self.interaction_graph_size = int(self.interaction_graph.shape[0] * self.interaction_graph.shape[1])
       
    def build_interaction_graph(self):
        """
        Build the interaction graph using the specified distance approach.

        Returns:
        - interaction_graph (numpy array): The interaction graph.
        - distances (numpy array): The pairwise distances between atoms.
        """
        conf1, coords1, atoms1 = self.get_atomic_coordinates_and_atoms(self.molecule1)
        conf2, coords2, atoms2 = self.get_atomic_coordinates_and_atoms(self.molecule2)

        if self.search_method == 'cdist':
            distances = cdist(coords1, coords2)
            interaction_graph = (distances <= self.distance_cutoff).astype(int)

        elif self.search_method == 'kdtree':
            kdtree = cKDTree(coords2)
            interaction_graph = np.zeros((len(coords1), len(coords2)), dtype=int)
            distances = np.full((len(coords1), len(coords2)), np.inf, dtype=float)

            for i, coord1 in enumerate(coords1):
                indices = kdtree.query_ball_point(coord1, self.distance_cutoff)
                for j in indices:
                    distance = np.linalg.norm(coord1 - coords2[j])
                    interaction_graph[i, j] = 1
                    distances[i, j] = distance

        elif self.search_method == 'balltree':
            ball_tree = BallTree(coords2)
            interaction_graph = np.zeros((len(coords1), len(coords2)), dtype=int)
            distances = np.full((len(coords1), len(coords2)), np.inf, dtype=float)

            for i, coord1 in enumerate(coords1):
                indices = ball_tree.query_radius([coord1], r=self.distance_cutoff)[0]
                for j in indices:
                    distance = np.linalg.norm(coord1 - coords2[j])
                    interaction_graph[i, j] = 1
                    distances[i, j] = distance

        else:
            raise ValueError("Invalid search_method. Choose 'cdist', 'kdtree', or 'balltree'.")

        if self.filter_obstructions:
            interaction_graph = self.filter_obstructed_interactions(interaction_graph, coords1, coords2, atoms1, atoms2)

        return interaction_graph, distances

    def build_batch_interaction_graphs(self, 
                                     coords1_batch: List[torch.Tensor], 
                                     coords2_batch: List[torch.Tensor],
                                     device: Optional[str] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Build interaction graphs for a batch of molecule pairs on GPU
        
        Args:
            coords1_batch: List of coordinate tensors for molecule 1 in each pair
            coords2_batch: List of coordinate tensors for molecule 2 in each pair
            device: Device to use (optional, defaults to self.device)
            
        Returns:
            Tuple of (interaction_graphs, distance_matrices)
        """
        if device is None:
            device = self.device
        else:
            device = torch.device(device)
            
        interaction_graphs = []
        distance_matrices = []
        
        for coords1, coords2 in zip(coords1_batch, coords2_batch):
            # Ensure inputs are tensors
            if not isinstance(coords1, torch.Tensor):
                coords1 = torch.tensor(coords1, dtype=torch.float32)
            if not isinstance(coords2, torch.Tensor):
                coords2 = torch.tensor(coords2, dtype=torch.float32)
                
            # Move to specified device
            coords1 = coords1.to(device)
            coords2 = coords2.to(device)
            
            # Compute pairwise distances using broadcasting
            # coords1: [N1, 3], coords2: [N2, 3]
            # distances: [N1, N2]
            distances = torch.cdist(coords1.unsqueeze(0), coords2.unsqueeze(0)).squeeze(0)
            
            # Create interaction graph
            interaction_graph = (distances <= self.distance_cutoff).float()
            
            interaction_graphs.append(interaction_graph)
            distance_matrices.append(distances)
        
        return interaction_graphs, distance_matrices
    
    def reduce_interaction_graphs_batch(self, 
                                      interaction_graphs: List[torch.Tensor],
                                      distance_matrices: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[List[int]], List[List[int]], List[torch.Tensor]]:
        """
        Reduce interaction graphs to only include interacting atoms (batch version)
        
        Args:
            interaction_graphs: List of interaction graph tensors
            distance_matrices: List of distance matrix tensors
            
        Returns:
            Tuple of (reduced_graphs, interacting_atoms1_batch, interacting_atoms2_batch, reduced_distances)
        """
        reduced_graphs = []
        interacting_atoms1_batch = []
        interacting_atoms2_batch = []
        reduced_distances = []
        
        for interaction_graph, distances in zip(interaction_graphs, distance_matrices):
            # Find interacting atoms
            interacting_atoms1 = torch.where(interaction_graph.sum(dim=1) > 0)[0]
            interacting_atoms2 = torch.where(interaction_graph.sum(dim=0) > 0)[0]
            
            # Reduce graphs and distances
            reduced_graph = interaction_graph[interacting_atoms1][:, interacting_atoms2]
            reduced_dist = distances[interacting_atoms1][:, interacting_atoms2]
            
            reduced_graphs.append(reduced_graph)
            interacting_atoms1_batch.append(interacting_atoms1.cpu().tolist())
            interacting_atoms2_batch.append(interacting_atoms2.cpu().tolist())
            reduced_distances.append(reduced_dist)
        
        return reduced_graphs, interacting_atoms1_batch, interacting_atoms2_batch, reduced_distances
       
    def reduce_interaction_graph(self, interaction_graph, distances):
        """
        Reduce the interaction graph and distances by keeping only atoms that interact with at least one atom in the other molecule.

        Parameters:
        - interaction_graph (numpy array): The full interaction graph.
        - distances (numpy array): The full pairwise distances matrix.

        Returns:
        - reduced_graph (torch.Tensor): The reduced interaction graph.
        - interacting_atoms1 (list): Indices of interacting atoms in molecule1.
        - interacting_atoms2 (list): Indices of interacting atoms in molecule2.
        - reduced_distances (torch.Tensor): Reduced pairwise distances corresponding to the reduced interaction graph.
        """

        if self.debug:
            print(f"Number of atoms in molecule1: {len(self.molecule1.GetAtoms())}")
            print(f"Number of atoms in molecule2: {len(self.molecule2.GetAtoms())}")
            print(f"Interaction graph size: {interaction_graph.shape}")
            print(f"Distances size: {distances.shape}")

        # Identify interacting atoms in molecule1 and molecule2
        interacting_atoms1 = np.where(np.any(interaction_graph, axis=1))[0].tolist()
        interacting_atoms2 = np.where(np.any(interaction_graph, axis=0))[0].tolist()

        # Subset the interaction graph and distances
        reduced_graph = interaction_graph[interacting_atoms1][:, interacting_atoms2]
        reduced_distances = distances[interacting_atoms1][:, interacting_atoms2]

        if self.debug:
            print(f"Number of interacting atoms in molecule1: {len(interacting_atoms1)}")
            print(f"Number of interacting atoms in molecule2: {len(interacting_atoms2)}")
            print(f"Reduced interaction graph size: {reduced_graph.shape}")
            print(f"Reduced distances size: {reduced_distances.shape}")

        return (torch.tensor(reduced_graph, dtype=torch.float32), 
                interacting_atoms1, 
                interacting_atoms2, 
                torch.tensor(reduced_distances, dtype=torch.float32))

    def filter_obstructed_interactions(self, interaction_graph, coords1, coords2, atoms1, atoms2):
        """
        Remove interactions from the interaction graph where the interaction is obstructed by another atom.
        """
        obstructed_graph = np.copy(interaction_graph)
        indices = np.argwhere(interaction_graph == 1)

        if indices.size == 0:
            return obstructed_graph

        all_coords = np.concatenate((coords1, coords2))
        all_atoms = np.concatenate((atoms1, atoms2))

        for idx in indices:
            i, j = idx
            point1 = coords1[i]
            point2 = coords2[j]
            if self.is_obstructed(point1, point2, all_coords, all_atoms):
                obstructed_graph[i, j] = 0

        return obstructed_graph

    def is_obstructed(self, point1, point2, all_coords, all_atoms):
        """
        Vectorized check for obstruction between point1 and point2.
        """
        line_vec = point2 - point1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return False
        line_unitvec = line_vec / line_len

        points_vec = all_coords - point1
        projections = np.dot(points_vec, line_unitvec)

        # Mask for projections that are within the segment
        segment_mask = (projections > 0) & (projections < line_len)

        # Calculate the closest points on the line segment
        closest_points = point1 + np.outer(projections, line_unitvec)

        # Calculate distances to the line segment
        distances_to_line = np.linalg.norm(all_coords - closest_points, axis=1)

        # Get the Van der Waals radii for all atoms
        vdw_radii = np.array([self.get_vdw_radius(atom) for atom in all_atoms])

        # Mask for distances that are within the vdw radius accounting for tolerance
        obstruction_mask = (distances_to_line < (vdw_radii - self.obstruction_tolerance)) & segment_mask

        # Check if any point obstructs
        return np.any(obstruction_mask)

    def get_vdw_radius(self, atom):
        """
        Get the van der Waals radius of an atom using RDKit.
        """
        return Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())

    def get_atomic_coordinates_and_atoms(self, molecule):
        """
        Retrieve atomic coordinates and atom objects from the molecule.

        Parameters:
        - molecule (RDKit Mol or BioPython Structure): The molecule.

        Returns:
        - coords: A numpy array of 3D coordinate lists.
        - atoms: List of atom objects.
        """
        # Get the conformer
        conformer = self.get_conformer(molecule)

        if isinstance(molecule, Chem.Mol):
            coords = [conformer.GetAtomPosition(i) for i in range(molecule.GetNumAtoms())]
            coords = np.array([[coord.x, coord.y, coord.z] for coord in coords])
            atoms = [molecule.GetAtomWithIdx(i) for i in range(molecule.GetNumAtoms())]
        elif isinstance(molecule, Structure):
            coords = np.array([atom.coord for atom in molecule.get_atoms()])
            atoms = list(molecule.get_atoms())
        else:
            print(f"Unsupported molecule type: {type(molecule)}")
            print(f"Molecule info: {molecule}")
            raise ValueError("Invalid molecule type. Supported types: RDKit Mol, BioPython Structure.")

        return conformer, coords, atoms

    def get_conformer(self, molecule):
        """
        Get the conformer of the molecule.

        Parameters:
        - molecule: The input molecule (RDKit Mol or BioPython Structure).

        Returns:
        - conformer: The molecule conformer.
        """
        if isinstance(molecule, Chem.Mol):
            conformer = molecule.GetConformer()
        elif isinstance(molecule, Structure):
            conformer = molecule[0]  # Get just the first Model from the Structure
        else:
            print(f"Unsupported molecule type: {type(molecule)}")
            print(f"Molecule info: {molecule}")
            raise ValueError("Invalid molecule type. Supported types: RDKit Mol, BioPython Structure.")

        return conformer

    def get_interaction_molecule(self, molecule, interaction_atoms):
        """
        Get a new molecule containing only the atoms involved in interactions.

        Parameters:
        - molecule: The original molecule (RDKit Mol or BioPython Structure).
        - interaction_atoms: List of atom indices involved in interactions.

        Returns:
        - interaction_molecule: The new molecule containing only interacting atoms.
        """
        if isinstance(molecule, Chem.Mol):
            # Get the complement of interaction_atoms because we will be deleting the non-interacting atoms
            complement_atoms = [idx for idx in range(molecule.GetNumAtoms()) if idx not in interaction_atoms]
            interaction_molecule = Chem.RWMol(molecule)
            for atom_idx in reversed(complement_atoms):
                interaction_molecule.RemoveAtom(int(atom_idx))
            return interaction_molecule.GetMol()

        elif isinstance(molecule, Structure):
            # Extract coordinates and element symbols for selected atoms
            selected_atom_data = [(tuple(atom.get_coord()), atom.element) for i, atom in enumerate(molecule.get_atoms()) if i in interaction_atoms]

            # Create an RDKit molecule
            interaction_molecule = Chem.RWMol()  # Use RWMol for building the molecule

            # Add atoms and set their positions
            for coord, element in selected_atom_data:
                atom = Chem.Atom(element)
                interaction_molecule.AddAtom(atom)

            # Create a conformer and set atom positions
            conf = Chem.Conformer(len(selected_atom_data))
            for i, (coord, _) in enumerate(selected_atom_data):
                # Convert coordinates to RDKit Point3D type
                coord_3d = Point3D(float(coord[0]), float(coord[1]), float(coord[2]))
                conf.SetAtomPosition(i, coord_3d)
            interaction_molecule.AddConformer(conf)

            # Convert RWMol to RDKit Mol
            interaction_molecule = interaction_molecule.GetMol()

            return interaction_molecule

        else:
            raise ValueError("Invalid molecule type. Supported types: RDKit Mol, BioPython Structure.")

    def create_interaction_molecules(self):
        """
        Create new molecules based on the reduced interaction graph.

        Returns:
        - interaction_molecule1: New molecule containing only atoms involved in interactions from molecule1.
        - interaction_molecule2: New molecule containing only atoms involved in interactions from molecule2.
        """
        interaction_molecule1 = self.get_interaction_molecule(self.molecule1, self.interacting_atoms1)
        interaction_molecule2 = self.get_interaction_molecule(self.molecule2, self.interacting_atoms2)

        return interaction_molecule1, interaction_molecule2

    @staticmethod
    def get_coordinates_from_molecule(molecule):
        """
        Static method to extract coordinates from a molecule as a torch tensor.
        
        Parameters:
        - molecule (RDKit Mol or BioPython Structure): The molecule.
        
        Returns:
        - torch.Tensor: Coordinates as a tensor of shape [num_atoms, 3].
        """
        if isinstance(molecule, Chem.Mol):
            conformer = molecule.GetConformer()
            coords = [conformer.GetAtomPosition(i) for i in range(molecule.GetNumAtoms())]
            coords = np.array([[coord.x, coord.y, coord.z] for coord in coords])
        elif isinstance(molecule, Structure):
            coords = np.array([atom.coord for atom in molecule.get_atoms()])
        else:
            raise ValueError("Invalid molecule type. Supported types: RDKit Mol, BioPython Structure.")
        
        return torch.tensor(coords, dtype=torch.float32)

    def write_xyz_file(self, molecule, filename):
        """
        Write the molecule to an XYZ file.

        Parameters:
        - molecule: The molecule (RDKit Mol or Biopython Structure).
        - filename (str): The filename for the XYZ file.
        """
        if isinstance(molecule, Chem.Mol):
            Chem.MolToXYZFile(molecule, filename)
        elif isinstance(molecule, Structure):
            with open(filename, 'w') as f:
                atoms = Selection.unfold_entities(molecule, 'A')
                num_atoms = len(atoms)
                f.write(f"{num_atoms}\n")
                f.write("Comment line\n")  # You can modify this line as needed
                for atom in molecule.get_atoms():
                    symbol = atom.get_element()  # Updated to get the element symbol
                    coords = atom.get_coord()
                    f.write(f"{symbol} {coords[0]} {coords[1]} {coords[2]}\n")
        else:
            raise ValueError("Invalid molecule type. Supported types: RDKit Mol, BioPython Structure.")