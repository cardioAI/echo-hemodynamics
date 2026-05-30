import numpy as np

PARAM_NAMES = ["RAP", "SPAP", "dpap", "meanPAP", "PCWP", "CO", "CI", "SVRI", "PVR"]
PARAM_PALETTE = [
    (231 / 255, 98 / 255, 84 / 255),
    (239 / 255, 138 / 255, 71 / 255),
    (247 / 255, 170 / 255, 88 / 255),
    (255 / 255, 208 / 255, 111 / 255),
    (255 / 255, 230 / 255, 183 / 255),
    (170 / 255, 220 / 255, 224 / 255),
    (114 / 255, 188 / 255, 213 / 255),
    (82 / 255, 143 / 255, 173 / 255),
    (55 / 255, 103 / 255, 149 / 255),
]
CLINICAL_CUTOFFS = {
    "RAP": 8.0,
    "SPAP": 40.0,
    "dpap": 15.0,
    "meanPAP": 20.0,
    "PCWP": 15.0,
    "CO": 4.0,
    "CI": 2.5,
    "SVRI": 2400.0,
    "PVR": 2.0,
}
PARAM_UNITS = {
    "RAP": "mmHg",
    "SPAP": "mmHg",
    "dpap": "mmHg",
    "meanPAP": "mmHg",
    "PCWP": "mmHg",
    "CO": "L/min",
    "CI": "L/min/m2",
    "SVRI": "dyn-s/cm5",
    "PVR": "Wood Units",
}


def calculate_correlation(pred, target):
    if len(pred) != len(target):
        return 0.0
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()
    if np.std(pred) < 1e-08 or np.std(target) < 1e-08:
        return 0.0
    corr = np.corrcoef(pred, target)[0, 1]
    return abs(corr) if not np.isnan(corr) else 0.0
