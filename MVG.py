import numpy
import utils

def MVG_logLikelihoodRatios(DTR, LTR, DTE, params=None):
    # mean
    m = utils.means(DTR, LTR)
    # covariance matrix
    c = utils.covariances(DTR, LTR, m)
    
    # Compute the loglikelihood of each sample for each class 
    scores = utils.compute_loglikelihoods (DTE, m, c, 2)
    #return the ratio
    return (scores[1,:]-scores[0,:])

def TIED_logLikelihoodRatios(DTR, LTR, DTE, params=None):
    m = utils.means(DTR, LTR)
    c_MVG = utils.covariances(DTR, LTR, m)


    # The tied covariance is a covariance matrix in which the covariance between all pairs of features is assumed to be the same
    csigma = numpy.zeros([DTR.shape[0],DTR.shape[0]])

    for i in range (len(c_MVG)):
        Nc = (LTR == i).sum()
        csigma += c_MVG[i]*Nc
    csigma = csigma / LTR.shape[0]

    c =[]
    for i in range (len(c_MVG)):
        c.append(csigma)
    
    # Compute the loglikelihood of each sample for each class
    scores = utils.compute_loglikelihoods (DTE, m, c, 2)
    #return the ratio
    return (scores[1,:]-scores[0,:])

def NAIVE_logLikelihoodRatios(DTR, LTR, DTE,params=None):
    m = utils.means(DTR, LTR)
    c_MVG = utils.covariances(DTR, LTR, m)


    # Diagonal covariance matrix, take only the diagonal elemnets of covariance matrix
    c = c_MVG
    for i in range (len(c_MVG)):
        c[i]= c_MVG[i]*numpy.eye(c_MVG[i].shape[0],c_MVG[i].shape[1])
    
    # Compute the loglikelihood of each sample for each class
    scores = utils.compute_loglikelihoods (DTE, m, c, 2)
    #return the ratio
    return (scores[1,:]-scores[0,:])

def TIED_DIAG_COV_logLikelihoodRatios(DTR, LTR, DTE, params=None):
    m = utils.means(DTR, LTR)
    c_MVG = utils.covariances(DTR, LTR, m)

    csigma = numpy.zeros([DTR.shape[0], DTR.shape[0]])

    for i in range (len(c_MVG)):
        Nc = (LTR == i).sum()
        csigma += c_MVG[i]*Nc
    csigma = csigma / LTR.shape[0]

    # we need only the diagonal
    csigma= csigma*numpy.eye(csigma.shape[0],csigma.shape[1])

    c =[]
    for i in range (len(c_MVG)):
        c.append(csigma)
        
    
    # Compute the loglikelihood of each sample for each class
    scores = utils.compute_loglikelihoods (DTE, m, c, 2)
    #return the ratio
    return (scores[1,:]-scores[0,:])