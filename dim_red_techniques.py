import numpy
import scipy.linalg as linalg

def mcol(v):
    return v.reshape((v.size, 1))

def PCA(D, m):
    mu = D.mean(1)
    normalized_data = D - mcol(mu)
    # N = number of samples
    N = D.shape[1]
    #compute the covariance matrix
    covariance_matrix= numpy.dot(normalized_data, normalized_data.T) / N

    #eigenvectors and eigenvalues
    eigenvalues,eigenvectors = numpy.linalg.eigh(covariance_matrix)
    #retrieve the first m eigenvectors
    eigenvectors_selected = eigenvectors[:, ::-1][:,0:m]

    principal_components = numpy.dot(eigenvectors_selected.T, normalized_data)
    return principal_components
