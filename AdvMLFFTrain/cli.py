import logging
from AdvancedMLFFTrain.utils import parse_args
from AdvancedMLFFTrain.active_learning import ActiveLearning
from AdvancedMLFFTrain.data_reduction import DataReduction

def main():
    """Runs the selected pipeline based on user input."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.pipeline == "active_learning":
        pipeline = ActiveLearning(args)
    elif args.pipeline == "data_reduction":
        pipeline = DataReduction(args)

    pipeline.run()

if __name__ == "__main__":
    main()

