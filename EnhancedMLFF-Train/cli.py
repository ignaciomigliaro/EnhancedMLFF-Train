import argparse

def parse_args():
    """Parses command-line arguments for both active learning and data reduction."""
    parser = argparse.ArgumentParser(description="Machine Learning Force Field Training")

    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    # Active Learning
    al_parser = subparsers.add_parser("active_learning", help="Run active learning pipeline")
    al_parser.add_argument("--filepath", type=str, required=True)
    al_parser.add_argument("--stepsize", type=int, default=1)
    al_parser.add_argument("--model_dir", type=str, required=True)
    al_parser.add_argument("--calculator", type=str, required=True)
    al_parser.add_argument("--device", type=str, default="cpu")
    al_parser.add_argument("--threshold", type=float)
    al_parser.add_argument("--output_dir", type=str, default="results")

    # Data Reduction
    dr_parser = subparsers.add_parser("data_reduction", help="Run data reduction pipeline")
    dr_parser.add_argument("--filepath", type=str, required=True)
    dr_parser.add_argument("--stepsize", type=int, default=1)
    dr_parser.add_argument("--model_dir", type=str, required=True)
    dr_parser.add_argument("--calculator", type=str, required=True)
    dr_parser.add_argument("--device", type=str, default="cpu")
    dr_parser.add_argument("--output_dir", type=str, default="reduced_data")
    dr_parser.add_argument("--dft_energy_file", type=str, required=True, help="Path to DFT energy data")

    return parser.parse_args()
