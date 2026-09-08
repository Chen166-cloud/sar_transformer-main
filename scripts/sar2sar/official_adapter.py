# SPDX-License-Identifier: GPL-3.0
"""Normalization from the authors' SAR2SAR-test/utils.py and model.py.

E. Dalsasso, L. Denis, F. Tupin (2020), Telecom Paris, GPL-3.0.
The official package remains unchanged; these functions reproduce its exact
float32 casting, clipping and scaling order for current local inference.
"""

import numpy as np

M = 10.089038980848645
m = -1.429329123112601
OUTPUT_TENSOR = "sub:0"


def normalize(amplitude):
    return ((np.log(np.clip(amplitude, 0.24, np.max(amplitude))) - m) * 255 / (M - m)).astype("float32") / 255.0


def denormalize(normalized_log_amplitude):
    return np.exp((M - m) * np.clip(np.squeeze(normalized_log_amplitude).astype("float32"), 0, 1) + m)


NORMALIZATION = {
    "M": M, "m": m, "input_amplitude_floor": 0.24,
    "input_formula": "float32((log(clip(A,0.24,max(A)))-m)*255/(M-m))/255",
    "output_formula": "exp((M-m)*clip(float32(mean_patch_prediction),0,1)+m)",
    "patch_aggregation_domain": "normalized_log_amplitude_before_inverse_transform",
    "source": "SAR2SAR-test/utils.py:normalize_sar,denormalize_sar;model.py:denoiser.test",
}
