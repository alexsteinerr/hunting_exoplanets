# Configuration constants for multi-target analysis
SEED = 42

# Data settings
MISSION = "TESS"
USE_ALL_SECTORS = False
REMOVE_NANS = True
NORMALIZE = True

# Binning
USE_BINNING = True
NBINS = 400

# Model settings
H1, H2 = 256, 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 3000  # Further reduced for faster processing
PATIENCE = 150
ALPHA_TRANSIT = 10.0

# Analysis settings
HALF_WINDOW = 0.06
BASELINE = 1.0
DENSE_SAMPLES = 1000  # Reduced for faster processing
CHEB_DEGREE = 6

# Multi-target analysis
MAX_TARGETS = 200  # Set to None for all available targets
SKIP_FAILED = True
SAVE_INDIVIDUAL_RESULTS = True

# API Settings
USE_API = True
CACHE_TARGETS = True
MIN_PERIOD_DAYS = 0.3
MAX_PERIOD_DAYS = 50.0

# Output
RESULTS_CSV = "exoplanet_transit_areas.csv"
SUMMARY_CSV = "exoplanet_analysis_summary.csv"
TARGETS_CSV = "api_exoplanet_targets.csv"