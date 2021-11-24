"""Make a dataset from raw data."""
import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

SEED = 90210

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/"
    "master/palmerpenguins/data/penguins-raw.csv"
)

MAPPING = {
    "Culmen Length (mm)": "culmen_length_mm",
    "Culmen Depth (mm)": "culmen_depth_mm",
    "Flipper Length (mm)": "flipper_length_mm",
    "Body Mass (g)": "body_mass_g",
    "Species": "species",
}
TARGET = "species"
FEATURES = [feature for feature in MAPPING.values() if feature != TARGET]


@click.command()
@click.option(
    "--nr_lines", default=10, help="Number of lines to add to the raw dataset"
)  # type: ignore
@click.option(
    "--pipeline_name", default="penguin_local", help="Number of the pipeline"
)  # type: ignore
def get_raw_data_slice(nr_lines: int, pipeline_name: str) -> None:
    """Download `nr_of_lines` (more) of raw data."""
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]
    filename = RAW_DATA_URL.rsplit("/", 1)[-1]
    output_dir = project_dir / "data" / pipeline_name / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / filename
    df = pd.read_csv(RAW_DATA_URL)
    df = df.dropna()
    df = df[list(MAPPING.keys())]
    df = df.rename(columns=MAPPING)

    # deterministic shuffle
    df = df.sample(frac=1, random_state=SEED)

    # normalize features
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # encode target labels
    label_encoder = LabelEncoder()
    df[TARGET] = label_encoder.fit_transform(df[TARGET])

    if not output_filepath.exists():
        logger.info(f"Making first raw data set with {nr_lines} lines")
        df = df.head(nr_lines)
    else:
        with open(output_filepath) as f:
            current_nr_lines = len(f.readlines()) - 1  # remove 1 for the header
        new_nr_lines = current_nr_lines + nr_lines
        logger.info(
            f"Updating raw data set. Already found {current_nr_lines}, "
            f"adding {nr_lines} lines"
        )
        df = df.iloc[:new_nr_lines, :]
    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    get_raw_data_slice()
