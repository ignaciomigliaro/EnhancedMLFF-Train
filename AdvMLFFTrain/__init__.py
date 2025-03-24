# AdvMLFFTrain/__init__.py
import os

CONFIG = {
    "data_path": os.getenv("DATA_PATH", "./data")
}

print("Your Project Package Loaded!")
