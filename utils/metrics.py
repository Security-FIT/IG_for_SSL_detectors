from scipy.stats import norm
from sklearn.metrics import det_curve, DetCurveDisplay
import numpy as np
import matplotlib.pyplot as plt

def calculate_EER(labels, predictions) -> float:
    """
    Calculate the Equal Error Rate (EER) from the labels and predictions
    """
    fpr, fnr, thresholds = det_curve(labels, predictions, pos_label=1)

    # eer from fpr and fnr can differ a bit (its an approximation), so we compute both and take the average
    eer_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_fpr + eer_fnr) / 2

    return eer

def calculate_minDCF(labels, predictions, p_target=0.95, c_miss=1, c_fa=10) -> float:
    """
    Calculate the minimum Detection Cost Function (minDCF)
    """
    frr, far, thresholds = det_curve(labels, predictions, pos_label=1)

    c_det = c_miss * frr * p_target + c_fa * far * (1 - p_target)
    min_c_det = np.min(c_det)

    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    
    return min_dcf
