import argparse
import os
import logging
from ase.io import read
import numpy as np 
from ase import Atoms
import random

def parse_args():
    """Parses command-line arguments for both active learning and data reduction."""
    parser = argparse.ArgumentParser(description="Machine Learning Force Field Training")

    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    # Active Learning Pipeline
    al_parser = subparsers.add_parser("active_learning", help="Run active learning pipeline")
    al_parser.add_argument("--filepath", type=str, required=True)
    al_parser.add_argument("--model_dir", type=str, required=True)
    al_parser.add_argument("--calculator", type=str, required=True)
    al_parser.add_argument("--device", type=str, default="cpu")
    al_parser.add_argument("--threshold", type=float)
    al_parser.add_argument("--output_dir", type=str, default="results")
    al_parser.add_argument("--dft_software", type=str, choices=["qe", "orca"])
    al_parser.add_argument("--eval_criteria", type=str, choices=["energy", "forces"], default="energy")
    al_parser.add_argument("--use_cache", type=str)
    al_parser.add_argument("--plot_std_dev", action="store_true")
    al_parser.add_argument("--lower_threshold", type=float)

    # Data Reduction Pipeline
    dr_parser = subparsers.add_parser("data_reduction", help="Run data reduction pipeline")
    dr_parser.add_argument("--filepath", type=str, required=True)
    dr_parser.add_argument("--stepsize", type=int, default=1)
    dr_parser.add_argument("--model_dir", type=str, required=True)
    dr_parser.add_argument("--calculator", type=str, required=True)
    dr_parser.add_argument("--device", type=str, default="cpu")
    dr_parser.add_argument("--output_dir", type=str, default="reduced_data")
    dr_parser.add_argument("--use_cache", type=str)
    dr_parser.add_argument("--target_size", type=int, required=True)

    return parser.parse_args()

def get_configurations(filepath):
    """Reads configurations from input files, optionally including DFT energy data."""
    logging.info(f"Reading configurations from {filepath}...")

    if os.path.isfile(filepath):
        configurations = read(filepath, index=":")
    elif os.path.isdir(filepath):
        configurations = []
        for filename in os.listdir(filepath):
            file_path = os.path.join(filepath, filename)
            if os.path.isfile(file_path):
                configurations.extend(read(file_path, index=":"))
    else:
        raise ValueError(f"Invalid path: {filepath}")

    return configurations

def parse_orca_to_ase(file_path):
    """
    Parses an ORCA output file and returns an ASE Atoms object with:
    - Atomic symbols
    - Positions (Angstroms)
    - Forces (eV/Å) (without negation)
    - Total energy (eV)

    Parameters:
        file_path (str): Path to the ORCA output file.

    Returns:
        Atoms: ASE Atoms object with energy and forces.
    """
    energy = None
    forces = []
    positions = []
    elements = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Extract final total energy (Hartree)
        for line in lines:
            if 'FINAL SINGLE POINT ENERGY' in line:
                energy = float(line.split()[-1]) * 27.2114  # Convert to eV

        # Extract atomic positions & elements from "CARTESIAN COORDINATES"
        in_positions_section = False
        for i, line in enumerate(lines):
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                in_positions_section = True
                continue  # Skip header line

            if in_positions_section:
                if line.strip() == "":  # End of positions section
                    break
                data = line.split()
                if len(data) == 4:
                    elements.append(data[0])  # Store atomic symbol
                    positions.append([float(data[1]), float(data[2]), float(data[3])])  # Store atomic positions

        # Extract Cartesian gradients (forces in Hartree/Bohr)
        in_forces_section = False
        skip_lines = 2  # Number of lines to skip after "CARTESIAN GRADIENT"

        for line in lines:
            if "CARTESIAN GRADIENT" in line:
                in_forces_section = True
                skip_lines = 2  # Reset the skip counter
                continue

            if in_forces_section:
                if skip_lines > 0:
                    skip_lines -= 1  # Skip the next two lines
                    continue

                if line.strip() == "":  # Stop when a blank line is encountered
                    break
                
                data = line.split()
                if len(data) >= 6 and data[2] == ":":  # Ensure correct parsing
                    try:
                        forces.append([
                            float(data[3]),  # X force
                            float(data[4]),  # Y force
                            float(data[5])   # Z force
                        ])
                    except ValueError:
                        break  # Stop if we encounter non-numeric data

        # Convert lists to numpy arrays, ensuring they are never None
        positions = np.array(positions) if positions else np.array([])
        forces = np.array(forces) * 51.422 if forces else np.array([])  # Convert Hartree/Bohr to eV/Å
        # Debugging print: Check lengths
        print(f"Positions: {len(positions) if positions.size else 0} | Forces: {len(forces) if forces.size else 0}")

        # Validate if element and position lists are the same length
        if len(elements) != len(positions):
            raise ValueError(f"Mismatch in element count ({len(elements)}) and position count ({len(positions)}).")

        # Create ASE Atoms object
        atoms = Atoms(symbols=elements, positions=positions)

        # Attach energy and forces
        if energy is not None:
            atoms.info['energy'] = energy

        # Ensure forces are added even if there’s a mismatch
        if forces.size > 0:
            atoms.info['forces'] = forces
        else:
            print(f"Warning: No forces extracted or mismatch in count for {file_path}")

        return atoms
    
    def random_sampling(atoms_list, percentage):
        if not (0 <= percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100")
        num_to_sample = int(len(atoms_list) * (percentage / 100))
        sampled_atoms = random.sample(atoms_list, num_to_sample)
        remaining_atoms = [atom for atom in atoms_list if atom not in sampled_atoms]
        return sampled_atoms, remaining_atoms