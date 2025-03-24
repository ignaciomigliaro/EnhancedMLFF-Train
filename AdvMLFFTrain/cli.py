import logging
import sys
import os 
import ase

# Add the project root to sys.path so Python can find AdvMLFFTrain
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AdvMLFFTrain.utils import parse_args

def main():
    """Runs the selected pipeline based on user input."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.pipeline == "active_learning":
        from AdvMLFFTrain.active_learning import ActiveLearning
        pipeline = ActiveLearning(args)
    elif args.pipeline == "data_reduction":
        from AdvMLFFTrain.data_reduction import DataReduction
        pipeline = DataReduction(args)

    pipeline.run()

if __name__ == "__main__":
    main()

