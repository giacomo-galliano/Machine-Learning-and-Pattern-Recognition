import numpy
import scipy.stats

# z_normalization: centering and scaling to unit variance
def z_normalization(D):
    Dmean = D.mean(axis=1).reshape(-1,1)
    Dstd_dev = numpy.std(D,axis=1).reshape(-1,1)
    z_norm = (D - Dmean)/ Dstd_dev
    return z_norm

# 
def gaussianization(D):
    N = D.shape[1]
    r = numpy.zeros(D.shape)
    for k in range(D.shape[0]):
        featureVector = D[k,:]
        ranks = scipy.stats.rankdata(featureVector, method='min') -1
        r[k,:] = ranks

    r = (r + 1)/(N+2)
    gaussianizedFeatures = scipy.stats.norm.ppf(r)
    return gaussianizedFeatures

