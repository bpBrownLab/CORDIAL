#!/usr/bin/env python

"""Utility for loading small molecules from various formats."""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

class SDFLoader:
    def __init__(self, sdf_file):
        """Initialize the SDFLoader with the path to an SDF file."""
        self.molecule = self.load_sdf(sdf_file)

    @staticmethod
    def load_sdf(sdf_file, all_mols=False, mol_index=0):
        """Load a molecule from an SDF file using RDKit with improved robustness."""
        # Suppress RDKit logging temporarily
        RDLogger.DisableLog('rdApp.*')

        mol_supplier = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False, strictParsing=False)
        
        def sanitize_mol(mol):
            if mol is None:
                return None
            try:
                Chem.SanitizeMol(mol)
            except:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except:
                    try:
                        mol.UpdatePropertyCache(strict=False)
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
                    except:
                        mol = None
            return mol

        try:
            if all_mols:
                molecules = []
                for mol in mol_supplier:
                    sanitized_mol = sanitize_mol(mol)
                    if sanitized_mol is not None:
                        molecules.append(sanitized_mol)
                return molecules
            else:
                mol = mol_supplier[mol_index]
                return sanitize_mol(mol)
        finally:
            # Re-enable RDKit logging
            RDLogger.EnableLog('rdApp.*')

class Mol2Loader:
    def __init__(self, mol2_file):
        """Initialize the Mol2Loader with the path to a Mol2 file."""
        self.molecule = self.load_mol2(mol2_file)

    @staticmethod
    def load_mol2(mol2_file):
        """Load a molecule from a Mol2 file using RDKit."""
        molecule = Chem.MolFromMol2File(mol2_file)
        return molecule

class SMILESLoader:
    def __init__(self, smiles):
        """Initialize the SMILESLoader with a SMILES string."""
        self.molecule = self.load_smiles(smiles)

    @staticmethod
    def load_smiles(smiles):
        """Load a molecule from a SMILES string using RDKit."""
        molecule = Chem.MolFromSmiles(smiles)
        return molecule

class SMARTSLoader:
    def __init__(self, smarts):
        """Initialize the SMARTSLoader with a SMARTS string."""
        self.molecule = self.load_smarts(smarts)

    @staticmethod
    def load_smarts(smarts):
        """Load a molecule from a SMARTS string using RDKit."""
        molecule = Chem.MolFromSmarts(smarts)
        return molecule

class PDBLoader:
    def __init__(self, pdb_file):
        """Initialize the PDBLoader with the path to a PDB file."""
        self.molecule = self.load_pdb(pdb_file)

    @staticmethod
    def load_pdb(pdb_file):
        """Load a molecule from a PDB file using RDKit."""
        molecule = Chem.MolFromPDBFile(pdb_file, sanitize=True, removeHs=False)
        return molecule
