import argparse
import os
import logging
from ase.io import read

def parse_args():
    """Parses command-line arguments for both active learning and data reduction."""
    parser = argparse.ArgumentParser(description="Machine Learning Force Field Training")

    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    # Active Learning Pipeline
    al_parser = subparsers.add_parser("active_learning", help="Run active learning pipeline")
    al_parser.add_argument("--filepath", type=str, required=True)
    al_parser.add_argument("--stepsize", type=int, default=1)
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
