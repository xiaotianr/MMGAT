from scipy.stats import chi2_contingency
import numpy as np
from sklearn.metrics import mutual_info_score
#calculate the mutual informtion
def calc_MI(x, y, bins=10):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    # mi = 0.5 * g / c_xy.sum()
    return mi
