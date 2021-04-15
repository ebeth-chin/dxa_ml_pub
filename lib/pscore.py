from sklearn.metrics import make_scorer
import scipy as sp

def pcc (x,y):
    r = sp.stats.pearsonr(x,y)[0]
    return r
