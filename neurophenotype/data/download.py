"""
PhysioNet dataset downloader — replaces wget on Windows.

Usage:
    python data/download.py                     # download all configured datasets
    python data/download.py --dataset chbmit    # specific dataset only
    python data/download.py --list              # show what will be downloaded

Adding a new dataset:
    Add an entry to DATASETS below, then run this script.
    Manually downloaded files (e.g. from Kaggle, Zenodo, paper supplements)
    can just be dropped into data/public/<folder_name>/ directly.

Requirements:
    pip install wfdb
"""
import argparse
import wfdb
from pathlib import Path

OUT_DIR = Path(__file__).parent / "public"

# Add datasets here as you find them.
# For PhysioNet datasets: fill in physionet_name + records.
# For manually downloaded files: just drop them in data/public/ — no entry needed.
DATASETS = {
    "chbmit": {
        "physionet_name": "chbmit",
        "description": "CHB-MIT Scalp EEG Database — pediatric seizure recordings (Dravet-relevant)",
        "records": [
            "chb01/chb01_01",
            "chb01/chb01_02",
            "chb01/chb01_03",
        ],
    },
    # Add more as you find them, e.g.:
    # "some-dataset": {
    #     "physionet_name": "physionet-slug",
    #     "description": "...",
    #     "records": ["subj01/record01"],
    # },
}


def download(dataset_key: str):
    cfg = DATASETS[dataset_key]
    dest = OUT_DIR / dataset_key
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {cfg['description']}")
    print(f"Destination: {dest}")
    print(f"Records    : {cfg['records']}\n")

    for record_name in cfg["records"]:
        print(f"  → {record_name}")
        try:
            wfdb.dl_files(
                db=cfg["physionet_name"],
                dl_dir=str(dest),
                files=[f"{record_name}.edf" if "/" in record_name else record_name],
            )
        except Exception as e:
            print(f"    [warn] {e} — trying wfdb.dl_database fallback")
            wfdb.dl_database(
                cfg["physionet_name"],
                dl_dir=str(dest),
                records=[record_name],
            )

    print(f"\nDone. Files in: {dest}")


def list_datasets():
    print("\nConfigured datasets:")
    for key, cfg in DATASETS.items():
        print(f"  {key:12s} — {cfg['description']}")
        for r in cfg["records"]:
            print(f"               {r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Download one dataset")
    parser.add_argument("--list", action="store_true", help="List configured datasets")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.dataset:
        download(args.dataset)
    else:
        for key in DATASETS:
            download(key)
