"""Test cases for make_dataset.py"""
from pathlib import Path
from unittest import mock

import pytest
from requests import exceptions as request_exception
from spolm.data import make_dataset


def test_get_raw_data(tmp_path):
    output_path = make_dataset.get_raw_data(pipeline_name="test", project_dir=tmp_path)
    testdata_path = (Path(__file__).parent / "testdata/iris.csv").resolve()
    assert output_path.open().read().strip() == testdata_path.open().read().strip()
    assert len(list(tmp_path.iterdir())) == 1


def test_get_raw_data_httperror(tmp_path):
    with mock.patch(
        "requests.get", side_effect=request_exception.HTTPError("Failed Request")
    ) as mock_request_post:
        with pytest.raises(SystemExit):
            output_path = make_dataset.get_raw_data(
                pipeline_name="test", project_dir=tmp_path
            )
            testdata_path = (Path(__file__).parent / "testdata/iris.csv").resolve()
            assert output_path.open().read() == testdata_path.open().read()
            assert len(list(tmp_path.iterdir())) == 1


def test_preprocess_data(tmp_path):
    output_path = make_dataset.get_raw_data(pipeline_name="test", project_dir=tmp_path)
    preprocessd_path = make_dataset.preprocess_data(output_path)
    assert (
        preprocessd_path.open().read()
        == """sepal.length,sepal.width,petal.length,petal.width,variety
0.0952380952380949,1.0,0.0,0.0,0
0.0,0.375,0.0,0.0,0
0.4285714285714284,0.0,0.8043478260869565,0.7391304347826086,2
1.0,0.625,0.7173913043478262,0.5217391304347826,1
0.6666666666666665,0.75,1.0,1.0000000000000002,2
0.7142857142857144,0.625,0.673913043478261,0.5652173913043478,1
"""
    )
    assert len(list(tmp_path.iterdir())) == 1
