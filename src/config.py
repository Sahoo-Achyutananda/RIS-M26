"""
Per-dataset configuration for the APT-detection replication pipeline.

Each entry tells the dataset-agnostic pipeline scripts (preprocess.py, balance.py,
feature_selection.py, ...) how to load a given dataset's raw files and turn them
into the common schema: a numeric feature matrix + a binary 'label' column
(0 = Normal/Benign, 1 = Attack/Anomaly).
"""

import glob
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.environ.get("RIS_DATA_RAW", os.path.join(PROJECT_ROOT, "data", "raw"))
DATA_PROCESSED = os.environ.get("RIS_DATA_PROCESSED", os.path.join(PROJECT_ROOT, "data", "processed"))
RESULTS_DIR = os.environ.get("RIS_RESULTS", os.path.join(PROJECT_ROOT, "results"))

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label",
]

DATASETS = {
    "nsl_kdd": {
        "display_name": "NSL-KDD",
        "raw_files": [
            os.path.join(DATA_RAW, "nsl_kdd", "KDDTrain+.txt"),
            os.path.join(DATA_RAW, "nsl_kdd", "KDDTest+.txt"),
        ],
        "has_header": False,
        "column_names": NSL_KDD_COLUMNS,
        "label_column": "label",
        "benign_values": {"normal"},
        "categorical_columns": ["protocol_type", "service", "flag"],
        "drop_columns": [],
        "n_features_to_select": 30,
        "sample_fraction": None,  # small enough already, no sampling needed
    },
    "unsw_nb15": {
        "display_name": "UNSW-NB15",
        "raw_files": [
            os.path.join(DATA_RAW, "unsw_nb15", "UNSW_NB15_training-set.csv"),
            os.path.join(DATA_RAW, "unsw_nb15", "UNSW_NB15_testing-set.csv"),
        ],
        "has_header": True,
        "encoding": "utf-8-sig",
        "column_names": None,
        "label_column": "label",
        "benign_values": {"0"},
        "categorical_columns": ["proto", "service", "state"],
        "drop_columns": ["id", "attack_cat"],
        "n_features_to_select": 30,
        "sample_fraction": None,
    },
    "cic_ids2017": {
        "display_name": "CIC-IDS2017",
        "raw_files": sorted(glob.glob(os.path.join(DATA_RAW, "cic_ids2017", "*.csv"))),
        "has_header": True,
        "encoding": "latin1",
        "column_names": None,
        "label_column": "Label",
        "benign_values": {"benign"},
        "categorical_columns": [],
        "drop_columns": ["Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP", "Timestamp"],
        "n_features_to_select": 30,
        "sample_fraction": None,
        "per_file_sample_fraction": None,  # files are small enough (~200MB each) to use in full
    },
    "cse_cic_ids2018": {
        "display_name": "CSE-CIC-IDS2018",
        "raw_files": sorted(glob.glob(os.path.join(DATA_RAW, "cse_cic_ids2018", "*.csv"))),
        "has_header": True,
        "encoding": "latin1",
        "column_names": None,
        "label_column": "Label",
        "benign_values": {"benign"},
        "categorical_columns": [],
        "drop_columns": ["Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port"],
        "n_features_to_select": 30,
        "sample_fraction": None,
        # Paper's own methodology: 0.2% stratified sample per day-file before
        # merging, since the full dataset is 16M+ rows across 10 files.
        "per_file_sample_fraction": 0.002,
    },
}


def get_config(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]
