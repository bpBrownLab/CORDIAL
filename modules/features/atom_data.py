#!/usr/bin/env python

from rdkit import Chem

hybridization = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 2.0,
    Chem.rdchem.HybridizationType.S: 3.0,
    Chem.rdchem.HybridizationType.SP: 5.0,
    Chem.rdchem.HybridizationType.SP2: 7.0,
    Chem.rdchem.HybridizationType.SP3: 11.0,
    Chem.rdchem.HybridizationType.SP2D: 13.0,
    Chem.rdchem.HybridizationType.SP3D: 17.0,
    Chem.rdchem.HybridizationType.SP3D2: 19.0,
    Chem.rdchem.HybridizationType.OTHER: 23.0
}

pauling_electronegativity = {
    1: 2.20,  # Hydrogen
    2: 0.00,  # None,  # Helium (Electronegativity not applicable)
    3: 0.98,  # Lithium
    4: 1.57,  # Beryllium
    5: 2.04,  # Boron
    6: 2.55,  # Carbon
    7: 3.04,  # Nitrogen
    8: 3.44,  # Oxygen
    9: 3.98,  # Fluorine
    10: 0.00, # None, # Neon (Electronegativity not applicable)
    11: 0.93, # Sodium
    12: 1.31, # Magnesium
    13: 1.61, # Aluminum
    14: 1.90, # Silicon
    15: 2.19, # Phosphorus
    16: 2.58, # Sulfur
    17: 3.16, # Chlorine
    18: 0.00, # None, # Argon (Electronegativity not applicable)
    19: 0.82, # Potassium
    20: 1.00, # Calcium
    21: 1.36, # Scandium
    22: 1.54, # Titanium
    23: 1.63, # Vanadium
    24: 1.66, # Chromium
    25: 1.55, # Manganese
    26: 1.83, # Iron
    27: 1.88, # Cobalt
    28: 1.91, # Nickel
    29: 1.90, # Copper
    30: 1.65, # Zinc
    31: 1.81, # Gallium
    32: 2.01, # Germanium
    33: 2.18, # Arsenic
    34: 2.55, # Selenium
    35: 2.96, # Bromine
    36: 3.00, # Krypton
    37: 0.82, # Rubidium
    38: 0.95, # Strontium
    39: 1.22, # Yttrium
    40: 1.33, # Zirconium
    41: 1.60, # Niobium
    42: 2.16, # Molybdenum
    43: 1.90, # Technetium
    44: 2.20, # Ruthenium
    45: 2.28, # Rhodium
    46: 2.20, # Palladium
    47: 1.93, # Silver
    48: 1.69, # Cadmium
    49: 1.78, # Indium
    50: 1.96, # Tin
    51: 2.05, # Antimony
    52: 2.10, # Tellurium
    53: 2.66, # Iodine
    54: 2.60, # Xenon
    55: 0.79, # Cesium
    56: 0.89, # Barium
    57: 1.10, # Lanthanum
    58: 1.12, # Cerium
    59: 1.13, # Praseodymium
    60: 1.14, # Neodymium
    61: 1.13, # Promethium
    62: 1.17, # Samarium
    63: 1.20, # Europium
    64: 1.20, # Gadolinium
    65: 1.10, # Terbium
    66: 1.22, # Dysprosium
    67: 1.23, # Holmium
    68: 1.24, # Erbium
    69: 1.25, # Thulium
    70: 1.25, # Ytterbium
    71: 1.27, # Lutetium
    72: 1.30, # Hafnium
    73: 1.50, # Tantalum
    74: 2.36, # Tungsten
    75: 1.90, # Rhenium
    76: 2.20, # Osmium
    77: 2.20, # Iridium
    78: 2.28, # Platinum
    79: 2.54, # Gold
    80: 2.00, # Mercury
    81: 1.62, # Thallium
    82: 2.33, # Lead
    83: 2.02, # Bismuth
    84: 2.00, # Polonium
    85: 2.20, # Astatine
    86: 0.00, # None, # Radon (Electronegativity not applicable)
    87: 0.70, # Francium
    88: 0.89, # Radium
    89: 1.10, # Actinium
    90: 1.30, # Thorium
    91: 1.50, # Protactinium
    92: 1.38, # Uranium
    93: 1.36, # Neptunium
    94: 1.28, # Plutonium
    95: 1.30, # Americium
    96: 1.30, # Curium
    97: 1.30, # Berkelium
    98: 1.30, # Californium
    99: 1.30, # Einsteinium
    100: 1.30, # Fermium
    101: 1.30, # Mendelevium
    102: 1.30, # Nobelium
    103: 1.30, # Lawrencium
}

# Empirical atomic polarizabilities in Å³ from Choudhary et al. 2019
# https://journals.sagepub.com/doi/10.1177/1747519819889936
polarizabilities = {
    1: 0.09, 2: 0.02, 3: 9.10, 4: 1.34, 5: 1.51, 6: 0.73, 7: 0.52, 8: 0.54,
    9: 0.46, 10: 0.43, 11: 11.80, 12: 3.18, 13: 5.23, 14: 2.19, 15: 1.34,
    16: 1.30, 17: 1.02, 18: 0.91, 19: 19.69, 20: 6.22, 21: 5.09, 22: 4.49,
    23: 4.48, 24: 4.33, 25: 3.48, 26: 3.07, 27: 3.01, 28: 3.12, 29: 3.01,
    30: 2.24, 31: 4.47, 32: 2.55, 33: 1.97, 34: 1.90, 35: 1.67, 36: 1.56,
    37: 22.14, 38: 7.60, 39: 5.57, 40: 4.63, 41: 4.45, 42: 4.08, 43: 3.80,
    44: 3.64, 45: 3.49, 46: 2.87, 47: 3.29, 48: 2.54, 49: 5.01, 50: 3.02,
    51: 2.37, 52: 2.22, 53: 1.97, 54: 1.82, 55: 26.99, 56: 11.07, 57: 8.68,
    58: 10.35, 59: 9.92, 60: 9.38, 61: 8.97, 62: 8.62, 63: 8.38, 64: 7.78,
    65: 7.91, 66: 7.74, 67: 7.58, 68: 7.46, 69: 7.35, 70: 7.26, 71: 7.73,
    72: 3.19, 73: 2.47, 74: 2.40, 75: 2.40, 76: 2.15, 77: 2.05, 78: 2.05,
    79: 2.00, 80: 1.85, 81: 7.00, 82: 4.16, 83: 4.04, 84: 2.98, 85: 2.40,
    86: 2.11, 87: 22.05, 88: 10.73, 89: 10.86, 90: 9.63, 91: 8.85, 92: 7.88,
    93: 7.46, 94: 7.30, 95: 7.18, 96: 7.11, 97: 6.65, 98: 6.48, 99: 6.31,
    100: 6.18, 101: 6.07, 102: 5.98, 103: 8.13}

# Absolute Hardness (η) in electron volts (eV) from various literature sources
# including Pearson, R. G. "Absolute electronegativity and hardness: application to inorganic chemistry", 1988.
hardness = {
    1: 7.20,   # Hydrogen
    2: 8.50,   # Helium
    3: 3.50,   # Lithium
    4: 5.00,   # Beryllium
    5: 4.50,   # Boron
    6: 5.30,   # Carbon
    7: 6.80,   # Nitrogen
    8: 8.80,   # Oxygen
    9: 10.30,  # Fluorine
    10: 11.00, # Neon
    11: 3.00,  # Sodium
    12: 3.50,  # Magnesium
    13: 3.60,  # Aluminum
    14: 4.00,  # Silicon
    15: 5.00,  # Phosphorus
    16: 6.00,  # Sulfur
    17: 6.10,  # Chlorine
    18: 6.50,  # Argon
    19: 2.50,  # Potassium
    20: 2.60,  # Calcium
    21: 3.20,  # Scandium
    22: 3.50,  # Titanium
    23: 3.50,  # Vanadium
    24: 3.65,  # Chromium
    25: 4.00,  # Manganese
    26: 4.20,  # Iron
    27: 4.20,  # Cobalt
    28: 4.30,  # Nickel
    29: 4.00,  # Copper
    30: 3.80,  # Zinc
    31: 3.20,  # Gallium
    32: 3.50,  # Germanium
    33: 3.80,  # Arsenic
    34: 4.00,  # Selenium
    35: 5.10,  # Bromine
    36: 5.50,  # Krypton
    37: 2.30,  # Rubidium
    38: 2.50,  # Strontium
    39: 3.00,  # Yttrium
    40: 3.20,  # Zirconium
    41: 3.60,  # Niobium
    42: 3.80,  # Molybdenum
    43: 4.00,  # Technetium
    44: 4.00,  # Ruthenium
    45: 4.10,  # Rhodium
    46: 4.10,  # Palladium
    47: 4.20,  # Silver
    48: 3.50,  # Cadmium
    49: 3.80,  # Indium
    50: 3.90,  # Tin
    51: 4.00,  # Antimony
    52: 4.20,  # Tellurium
    53: 4.90,  # Iodine
    54: 5.20,  # Xenon
    55: 2.10,  # Cesium
    56: 2.50,  # Barium
    57: 3.00,  # Lanthanum
    58: 3.20,  # Cerium
    59: 3.30,  # Praseodymium
    60: 3.40,  # Neodymium
    61: 3.50,  # Promethium
    62: 3.60,  # Samarium
    63: 3.70,  # Europium
    64: 3.80,  # Gadolinium
    65: 3.90,  # Terbium
    66: 4.00,  # Dysprosium
    67: 4.10,  # Holmium
    68: 4.20,  # Erbium
    69: 4.30,  # Thulium
    70: 4.40,  # Ytterbium
    71: 4.50,  # Lutetium
    72: 4.60,  # Hafnium
    73: 4.70,  # Tantalum
    74: 4.80,  # Tungsten
    75: 4.90,  # Rhenium
    76: 5.00,  # Osmium
    77: 5.10,  # Iridium
    78: 5.20,  # Platinum
    79: 5.30,  # Gold
    80: 5.40,  # Mercury
    81: 5.50,  # Thallium
    82: 5.60,  # Lead
    83: 5.70,  # Bismuth
    84: 5.80,  # Polonium
    85: 5.90,  # Astatine
    86: 6.00,  # Radon
    87: 2.00,  # Francium
    88: 2.50,  # Radium
    89: 3.00,  # Actinium
    90: 3.20,  # Thorium
    91: 3.40,  # Protactinium
    92: 3.60,  # Uranium
    93: 3.80,  # Neptunium
    94: 4.00,  # Plutonium
    95: 4.20,  # Americium
    96: 4.40,  # Curium
    97: 4.60,  # Berkelium
    98: 4.80,  # Californium
    99: 5.00,  # Einsteinium
    100: 5.20, # Fermium
    101: 5.40, # Mendelevium
    102: 5.60, # Nobelium
    103: 5.80  # Lawrencium
}
