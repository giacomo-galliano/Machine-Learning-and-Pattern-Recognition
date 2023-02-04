import numpy
import scipy.optimize

def mrow(v):
    return v.reshape((1,v.size)) #sizes gives the number of elements in the matrix/array

def mcol(v):
    return v.reshape((v.size, 1))

def logreg_obj_wrapper(DTR, LTR, lam, pi_T):

    def logreg_obj(v):
        w = v[0:-1]
        b = v[-1]
        # Zi for true class -> 1
        Zi_true=1
        # Zi for false class -> -1
        Zi_false=-1

        #num of samples belonging to the true class
        n_T = (1*(LTR==1)).sum() 
        #num of samples belonging to the false class
        n_F = (1*(LTR==0)).sum()

        data_true= DTR[:,LTR==1]
        data_false= DTR[:,LTR==0]

        func1 = (w*w).sum() * ( lam /2)

        #loss for class 1
        func2_1_true = numpy.dot( mrow(w.T),  data_true) + b
        func2_true= numpy.log1p(numpy.exp(- Zi_true * func2_1_true))

        #loss for class 0
        func2_1_false = numpy.dot( mrow(w.T),  data_false) + b
        func2_false= numpy.log1p(numpy.exp(- Zi_false * func2_1_false))


        func = func1 + (pi_T/n_T)* numpy.sum(func2_true, axis=1) + ((1-pi_T)/n_F)* numpy.sum(func2_false, axis=1)
        return func

    return logreg_obj

def LR_logLikelihoodRatios(DTR, LTR, DTE, params ):
    # compute the log likelihood ratios
    lam = params[0]
    pi_T = params[1]
    x0=numpy.zeros(DTR.shape[0]+1)
    logreg_obj = logreg_obj_wrapper(DTR, LTR, lam, pi_T)
    (x,f,d)= scipy.optimize.fmin_l_bfgs_b(logreg_obj, approx_grad=True, x0=x0, iprint=0 )
    
    w_min = mcol( x[0:-1] )
    b_min = x[-1]
    scores = numpy.dot( w_min.T , DTE ) + b_min
    return scores.flatten()

def expanded_data_elm(X):
    # feature expansion, can help to capture non-linear relationships between the features and the target variable.
    val = numpy.dot(X, X.T)
    v = mcol(val.flatten( 'F'))
    ex_X = numpy.vstack((v, X))
    return ex_X.flatten()

def expand_features (original_data):

    n_features = original_data.shape[0]
    ex_features_rows = n_features**2 + n_features
    ex_features_cols = original_data.shape[1]
    expanded_features = numpy.zeros((ex_features_rows, ex_features_cols))
    for i in range (ex_features_cols):
        expanded_features[:, i] = expanded_data_elm(mcol(original_data[:,i]))
    
    return expanded_features

def Quadratic_LR_logLikelihoodRatios(DTR, LTR, DTE, params ):
    expanded_DTR = expand_features(DTR)
    expanded_DTE = expand_features(DTE)

    return LR_logLikelihoodRatios(expanded_DTR, LTR, expanded_DTE, params )

