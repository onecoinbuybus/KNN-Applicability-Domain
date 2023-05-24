import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import DistanceMetric
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class AnomalyDetector:
    def __init__(self, random_state=444):
        self.random_state = random_state
        self.k = None
        self.dist_m = None
        self.d_wd = None
        self.ti = None
        self.Ki = None
    
    def calc_d_wd(self, dst_m, k):
        di_ave_vec = []
        for i in range(len(dst_m)):
            idx_list = [idx for idx in list(range(len(dst_m))) if idx !=i]
            di_ave = np.sort(dst_m[i][idx_list])[:k].mean()
            di_ave_vec.append(di_ave)  
        Q3d = np.percentile(di_ave_vec, 75)
        Q1d = np.percentile(di_ave_vec, 25) 
        d_wd = Q3d+1.5*(Q3d-Q1d)
        return d_wd

    def calc_ti_Ki(self, dst_m,d_wd,k):
        ti = []
        Ki = []
        for i in range(len(dst_m)):
            sorted_dst_v = np.sort(dst_m[i])[1:k+1]
            ad_inx = np.where(sorted_dst_v <= d_wd)
            ad_dist = sorted_dst_v[ad_inx]
            t = (np.sum(ad_dist))/(len(ad_dist))
            ti.append(t)
            Ki.append(len(ad_dist))
        return ti, Ki

    def compute_in_ad(self, x_train, x_test, ti):
        In_AD_idx = []
        which_train = []
        for i in range(len(x_test)):
            for j in range(len(x_train)):
                dis = distance.euclidean(x_train[j], x_test[i])
                if dis < ti[j]:
                    In_AD_idx.append(i)
                    which_train.append(j)
                    break
        return In_AD_idx, which_train

    def optimize_k(self, x):
        xidx = np.arange(len(x))
        train_idx, test_idx = train_test_split(xidx, test_size=0.25, random_state=self.random_state)
        x_train = x[train_idx]
        x_test = x[test_idx]
        dist = DistanceMetric.get_metric('euclidean')
        dist_m = dist.pairwise(x_train)
        In_AD_ratio = []
        for k in range(len(x_train)):
            d_wd = self.calc_d_wd(dist_m, k)
            ti, Ki = self.calc_ti_Ki(dist_m, d_wd, k)
            In_AD_idx, _ = self.compute_in_ad(x_train, x_test, ti)
            In_AD_ratio.append(len(In_AD_idx) / len(x_test))
        return In_AD_ratio

    def fit(self, x):
        self.x = x  
        In_AD_ratio_k = []
        for i in range(1000):
            In_AD_ratio_k.append(self.optimize_k(x))
        In_AD_ratio_k = np.array(In_AD_ratio_k)
        In_AD_ratios = []
        for i in range(In_AD_ratio_k.shape[1]):
            In_AD_ratios.append(np.median(In_AD_ratio_k[:, i]))
        self.k = In_AD_ratios.index(1.0) + 1
        dist = DistanceMetric.get_metric('euclidean')
        self.dist_m = dist.pairwise(x)
        self.d_wd = self.calc_d_wd(self.dist_m, self.k)
        self.ti, self.Ki = self.calc_ti_Ki(self.dist_m, self.d_wd, self.k)

    def predict(self, x_test):
        In_AD_idx, which_train = self.compute_in_ad(self.x, x_test, self.ti)
        labels = np.zeros(len(x_test))
        labels[In_AD_idx] = 1
        
        return labels, which_train

