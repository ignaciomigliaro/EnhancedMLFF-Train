import argparse

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
