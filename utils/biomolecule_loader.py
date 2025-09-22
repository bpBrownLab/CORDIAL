#!/usr/bin/env python

"""Utility for loading biomolecules (e.g., proteins) from various formats."""

from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

class PDBLoader:
    def __init__(self, pdb_file, structure_name='protein'):
        """
        Initialize the PDBLoader with the path to a PDB file.

        Parameters:
        - pdb_file (str): Path to the PDB file.
        - structure_name (str, optional): Name to assign to the loaded structure. Default is 'protein'.
        """
        self.structure = self.load_pdb(pdb_file, structure_name)

    def load_pdb(self, pdb_file, structure_name):
        """
        Load a structure from a PDB file using BioPython.

        Parameters:
        - pdb_file (str): Path to the PDB file.
        - structure_name (str): Name to assign to the loaded structure.

        Returns:
        - structure: Loaded structure.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(structure_name, pdb_file)
        return structure

class CIFLoader:
    def __init__(self, cif_file, structure_name='protein'):
        """
        Initialize the CIFLoader with the path to a CIF file.

        Parameters:
        - cif_file (str): Path to the CIF file.
        - structure_name (str, optional): Name to assign to the loaded structure. Default is 'protein'.
        """
        self.structure = self.load_cif(cif_file, structure_name)

    def load_cif(self, cif_file, structure_name):
        """
        Load a structure from a CIF file using BioPython.

        Parameters:
        - cif_file (str): Path to the CIF file.
        - structure_name (str): Name to assign to the loaded structure.

        Returns:
        - structure: Loaded structure.
        """
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(structure_name, cif_file)
        return structure

