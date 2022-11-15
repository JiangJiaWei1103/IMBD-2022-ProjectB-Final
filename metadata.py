"""
Project-specific metadata for global access.
Author: JiaWei Jiang
"""
# =Project B=
# Primary Key
PK = ["layer"]  # Normal dataset
PK_AUG = ["layer", "slice"]  # Mixed augmented dataset
# Features
SG_FEAT = ["b", "c", "d", "e", "f", "g", "h", "i"]  # SG
SPIKE_FEAT = ["b", "c", "d"]  # Spike
# Target
TARGET = "MaxWear"
TARGET2 = "MaxWearDiff"  # (Deprecated)

# Following "test"s are set to simulate final scenario
# Processing Layers
N_PROC_LAYERS = {"train1": 46, "train2": 24, "test": 46}  # Number of processing layers
GP1_LEN = {"train1": 26, "train2": 0, "test": 26}  # #Layers in group1
GP2_LEN = {"train1": 20, "train2": 0, "test": 20}  # #Layers in group2
# Number of Chunks per Layer
GP1_N_CHUNKS = {"train1": 11, "train2": 0, "test": 11}
GP2_N_CHUNKS = {"train1": 8, "train2": 0, "test": 8}
