import os
import logging
from ase.io import read

def get_configurations(filepath, stepsize=1, use_dft_energy=False, dft_energy_file=None):
    """Reads configurations from input files, optionally including DFT energy data."""
    logging.info(f"Reading configurations from {filepath}...")

    if os.path.isfile(filepath):
        configurations = read(filepath, index="::")[::stepsize]
    elif os.path.isdir(filepath):
        configurations = []
        for filename in os.listdir(filepath):
            file_path = os.path.join(filepath, filename)
            if os.path.isfile(file_path):
                configurations.extend(read(file_path, index="::")[::stepsize])
    else:
        raise ValueError(f"Invalid path: {filepath}")

    # If Data Reduction, get DFT energies
    if use_dft_energy and dft_energy_file:
        logging.info(f"Loading DFT energies from {dft_energy_file}")
        with open(dft_energy_file, "r") as f:
            dft_energies = {line.split()[0]: float(line.split()[1]) for line in f}
        for atoms in configurations:
            atoms.info["dft_energy"] = dft_energies.get(atoms.info.get("id"), None)

    return configurations
