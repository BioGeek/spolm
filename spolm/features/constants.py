from typing import Dict

RENAME = {
    "penguin": {
        "Culmen Length (mm)": "culmen_length_mm",
        "Culmen Depth (mm)": "culmen_depth_mm",
        "Flipper Length (mm)": "flipper_length_mm",
        "Body Mass (g)": "body_mass_g",
        "Species": "species",
    },
    "penguin_test": {
        "Culmen Length (mm)": "culmen_length_mm",
        "Culmen Depth (mm)": "culmen_depth_mm",
        "Flipper Length (mm)": "flipper_length_mm",
        "Body Mass (g)": "body_mass_g",
        "Species": "species",
    },
}

REPLACE = {
    "penguin": {
        "Adelie Penguin (Pygoscelis adeliae)": "Adelie",
        "Gentoo penguin (Pygoscelis papua)": "Gentoo",
        "Chinstrap penguin (Pygoscelis antarctica)": "Chinstrap",
    },
    "penguin": {
        "Adelie Penguin (Pygoscelis adeliae)": "Adelie",
        "Gentoo penguin (Pygoscelis papua)": "Gentoo",
        "Chinstrap penguin (Pygoscelis antarctica)": "Chinstrap",
    },
}

FEATURES = {
    "penguin": [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ],
    "penguin_test": [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ],
    "test": ["sepal.length", "sepal.width", "petal.length", "petal.width"],
}

TARGETS: Dict[str, str] = {
    "penguin": "species",
    "penguin_test": "species",
    "test": "variety",
}

RAW_DATA_URLS: Dict[str, str] = {
    "penguin": (
        "https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/"
        "master/palmerpenguins/data/penguins-raw.csv"
    ),
    "penguin_test": (
        "https://raw.githubusercontent.com/mcnakhaee/palmerpenguins/"
        "master/palmerpenguins/data/penguins-raw.csv"
    ),
    "test": (
        "https://raw.githubusercontent.com/BioGeek/"
        "spolm/main/tests/testdata/iris.csv"
    ),
}
