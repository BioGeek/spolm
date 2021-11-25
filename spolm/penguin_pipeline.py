"""TFX pipeline."""
from pathlib import Path
from typing import List

import absl
from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform
from tfx.orchestration import pipeline # type: ignore
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

_pipeline_name = "penguin_local"

_project_dir = Path(__file__).resolve().parents[2]
_raw_data_root = _project_dir / "data" / _pipeline_name / "raw"

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = _project_dir / "spolm" / "features" / "build_features.py"

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = _project_dir / "serving_model" / _pipeline_name

# Directory and data locations.
_tfx_root = _project_dir / "tfx"
_pipeline_root = _tfx_root / "pipelines" / _pipeline_name
_metadata_path = _tfx_root / "metadata" / _pipeline_name / "metadata.db"

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    "--direct_running_mode=multi_processing",
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    "--direct_num_workers=0",
]


def _create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_file: str,
    serving_model_dir: str,
    metadata_path: str,
    beam_pipeline_args: List[str],
) -> pipeline.Pipeline:
    """Implements the chicago taxi pipeline with TFX."""
    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = CsvExampleGen(input_base=data_root)
    print(f"{example_gen = }")

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    print(f"{statistics_gen = }")

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=False
    )
    print(f"{schema_gen = }")

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )
    print(f"{example_validator = }")

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
    )
    print(f"{transform = }")


# To run this pipeline from the python CLI:
#   $python pipeline.py
if __name__ == "__main__":
    absl.logging.set_verbosity(absl.logging.INFO)

    LocalDagRunner().run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=str(_pipeline_root),
            data_root=str(_raw_data_root),
            module_file=str(_module_file),
            serving_model_dir=str(_serving_model_dir),
            metadata_path=str(_metadata_path),
            beam_pipeline_args=_beam_pipeline_args,
        )
    )
