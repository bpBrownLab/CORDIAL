#!/usr/bin/env python

"""Module for loading molecules independent of format."""

from utils.molecule_loader import SDFLoader, PDBLoader
# from utils.biomolecule_loader import PDBLoader

def load_molecule(molecule_file):
	if molecule_file.endswith('.sdf'):
		sdf_loader = SDFLoader(molecule_file)
		return sdf_loader.molecule
	elif molecule_file.endswith('.pdb'):
		pdb_loader = PDBLoader(molecule_file)
		return pdb_loader.molecule
	else:
		raise ValueError(f"Unsupported file format: {molecule_file}")

def load_molecules( molecule_file):
	if molecule_file.endswith('.sdf'):
		molecules = SDFLoader.load_sdf(molecule_file, all_mols=True)
	return molecules
