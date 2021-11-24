"""Make a dataset from raw data."""
import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv
from dotenv import load_dotenv

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/"
    "master/palmerpenguins/data/penguins-raw.csv"
)


@click.command()
@click.option(
    "--nr_lines", default=10, help="Number of lines to add to the raw dataset"
)  # type: ignore
def get_raw_data_slice(nr_lines: int) -> None:
    """Download `nr_of_lines` (more) of raw data."""
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]
    filename = RAW_DATA_URL.rsplit("/", 1)[-1]
    output_filepath = project_dir / "data" / "raw" / filename
    df = pd.read_csv(RAW_DATA_URL)
    if not output_filepath.exists():
        logger.info(f"Making first raw data set with {nr_lines} lines")
        df = df.head(nr_lines)
        df.to_csv(output_filepath, index=False)
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
