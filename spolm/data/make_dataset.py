"""Make a dataset from raw data."""
import logging
from pathlib import Path

import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from spolm.features.constants import FEATURES
from spolm.features.constants import RAW_DATA_URLS
from spolm.features.constants import RENAME
from spolm.features.constants import REPLACE
from spolm.features.constants import TARGETS

SEED = 42


def get_raw_data(
    pipeline_name: str, project_dir: Path = Path(__file__).resolve().parents[2]
) -> Path:
    """Get raw data."""
    logger = logging.getLogger(__name__)
    url = RAW_DATA_URLS[pipeline_name]
    filename = url.rsplit("/", 1)[-1]
    output_dir = project_dir / "data" / pipeline_name / "unlabelled"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    logger.info(f"Downloading RAW data for pipeline {pipeline_name}")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    with open(output_path, "wb") as fp:
        fp.write(response.content)
    return output_path


def preprocess_data(raw_data_filepath: Path) -> Path:
    """Preprocess raw data."""
    logger = logging.getLogger(__name__)
    pipeline_name = raw_data_filepath.parts[-3]
    logger.info(f"Preprocessing RAW data for pipeline {pipeline_name}")
    df = pd.read_csv(raw_data_filepath)
    df = df.dropna()
    df = df.rename(columns=RENAME.get(pipeline_name, {}))
    df = df[FEATURES[pipeline_name] + [TARGETS[pipeline_name]]]
    df[TARGETS[pipeline_name]] = df[TARGETS[pipeline_name]].replace(
        REPLACE.get(pipeline_name, {})
    )
    # deterministic shuffle
    df = df.sample(frac=1, random_state=SEED)

    # normalize features
    scaler = MinMaxScaler()
    df[FEATURES[pipeline_name]] = scaler.fit_transform(df[FEATURES[pipeline_name]])

    # encode target labels
    label_encoder = LabelEncoder()
    df[TARGETS[pipeline_name]] = label_encoder.fit_transform(df[TARGETS[pipeline_name]])

    output_dir = Path(*(raw_data_filepath.parts[:-2] + ("labelled",)))
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = raw_data_filepath.parts[-1].replace("raw", "processed")
    output_path = output_dir / filename

    df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    raw_data_filepath = get_raw_data("penguin")
    preprocess_data(raw_data_filepath)
