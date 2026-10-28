"""
Functions and classes for generating specific types of model spectra.
"""

import numpy as np

__all__ = [
    "CLOUDY_LINE_MAP",
    "check_coverage",
    "NIRISS_050_FILTER_LIMITS",
    "NIRISS_001_FILTER_LIMITS",
]

CLOUDY_LINE_MAP = [
    # Paschen series
    {
        "cloudy": [
            "H  1  1.87510m",
        ],
        "grizli": "PaA",
        "wave": 18756.3,
    },
    {
        "cloudy": [
            "H  1  1.28181m",
        ],
        "grizli": "PaB",
        "wave": 12821.7,
    },
    {
        "cloudy": [
            "H  1  1.09381m",
        ],
        "grizli": "PaG",
        "wave": 10941.2,
    },
    {
        "cloudy": [
            "H  1  1.00494m",
        ],
        "grizli": "PaD",
        "wave": 10052.2,
    },
    # Balmer Series
    {
        "cloudy": [
            "H  1  6562.80A",
            "N  2  6583.45A",
            "N  2  6548.05A",
        ],
        "grizli": "Ha",
        "wave": 6564.697,
    },
    {
        "cloudy": ["H  1  4861.32A"],
        "grizli": "Hb",
        "wave": 4862.738,
    },
    # {
    #     "cloudy": ["H  1  4340.46A"],
    #     "grizli": "Hg",
    #     "wave": 4341.731,
    # },
    # {
    #     "cloudy": ["H  1  4101.73A"],
    #     "grizli": "Hd",
    #     "wave": 4102.936,
    # },
    # # Oxygen
    {
        "cloudy": [
            "O  3  5006.84A",
        ],
        "grizli": "OIII-5007",
        "wave": 5008.240,
    },
    {
        "cloudy": [
            "O  3  4958.91A",
        ],
        "grizli": "OIII-4959",
        "wave": 4960.295,
    },
    {
        "cloudy": ["O  3  4363.21A"],
        "grizli": "OIII-4363",
        "wave": 4364.436,
    },
    {
        "cloudy": [
            "O  2  3726.03A",
            "O  2  3728.81A",
        ],
        "grizli": "OII",
        "wave": 3728.48,
    },
    # Sulphur
    {
        "cloudy": [
            "S  3  9530.62A",
        ],
        "grizli": "SIII-9530",
        "wave": 9530.62,
    },
    {
        "cloudy": [
            "S  3  9068.62A",
        ],
        "grizli": "SIII-9068",
        "wave": 9068.62,
    },
    {
        "cloudy": [
            "S  2  6730.82A",
            "S  2  6716.44A",
        ],
        "grizli": "SII",
        "wave": 6725.48,
    },
    {
        "cloudy": [
            "S  3  6312.06A",
        ],
        "grizli": "SIII-6314",
        "wave": 6313.81,
    },
    # Helium
    {
        "cloudy": [
            "Blnd  1.08302m",
        ],
        "grizli": "HeI-1083",
        "wave": 10830.3,
    },
    {
        "cloudy": [
            "Blnd  5875.66A",
        ],
        "grizli": "HeI-5877",
        "wave": 5877.249,
    },
    {
        "cloudy": [
            "He 1  3888.64A",
        ],
        "grizli": "HeI-3889",
        "wave": 3889.75,
    },
    {
        "cloudy": [
            "Ne 3  3868.76A",
        ],
        "grizli": "NeIII-3867",
        "wave": 3869.87,
    },
]

# In Angstroms
NIRISS_050_FILTER_LIMITS = {
    # "F090W": [7960, 10050],
    "F115W": [10130, 12830],
    "F150W": [13300, 16710],
    "F200W": [17510, 22260],
}

NIRISS_001_FILTER_LIMITS = {
    # "F090W": [7960, 10050],
    "F115W": [10010, 12930],
    "F150W": [13200, 16810],
    "F200W": [17380, 22420],
}


def check_coverage(
    obs_wavelength: float, filter_limits: dict = NIRISS_001_FILTER_LIMITS
):
    """
    Check if a line is covered by the NIRISS filters.

    Parameters
    ----------
    obs_wavelength : float
        The observed wavelength of the line.
    filter_limits : dict, optional
        A dictionary, where the keys are the names of the grism filters,
        and the values are array-like, containing
        ``[min_wavelength, max_wavelength]``. Defaults to
        ``NIRISS_FILTER_LIMITS``.

    Returns
    -------
    bool
        ``True`` if the line falls within the filter coverage, else
        ``False``.
    """

    obs_wavelength = np.atleast_1d(obs_wavelength)
    covered = np.zeros((len(filter_limits), *obs_wavelength.shape), dtype=bool)
    for i, (k, v) in enumerate(filter_limits.items()):
        covered[i] = (obs_wavelength >= v[0]) & (obs_wavelength <= v[-1])
    return np.bitwise_or.reduce(covered, axis=0)
