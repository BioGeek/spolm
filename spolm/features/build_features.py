"""Script to turn raw data into features for modeling."""
import tensorflow_transform as tft

FEATURE_KEYS = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
_LABEL_KEY = "species"


def transformed_name(key):
    """Transform a key."""
    return key + "_xf"


# TFX Transform will call this function.
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in FEATURE_KEYS:
        # tft.scale_to_z_score computes the mean and variance of the given feature
        # and scales the output based on the result.
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])
    # Do not apply label transformation as it will result in wrong evaluation.
    outputs[transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

    return outputs
