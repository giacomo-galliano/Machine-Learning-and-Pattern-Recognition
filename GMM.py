import numpy
import scipy.special

def mcol(v):
    return v.reshape((v.size, 1))
def mrow(v):
    return v.reshape((1, v.size))

def logpdf_GAU_ND(x, mu, C):
    # logarithmic values of the probability density function
    M = x.shape[0]
    _, det = numpy.linalg.slogdet(C)
    det = numpy.log(numpy.linalg.det(C))
    inv = numpy.linalg.inv(C)
    
    res = []
    x_centered = x - mu
    for x_col in x_centered.T:
        res.append(numpy.dot(x_col.T, numpy.dot(inv, x_col)))

    return -M/2*numpy.log(2*numpy.pi) - 1/2*det - 1/2*numpy.hstack(res).flatten()

def logpdf_GMM(X, GMM):
    #x = samples, matrix of shape (D=size of a sample, N= number of samples)
    # gmm = [(w1,mu1, C1), (w2,mu2,C2),...]
    S= compute_joint_log_densities(X,GMM)
    
    #compute log marginal log density
    marginal_logdens = scipy.special.logsumexp(S, axis=0) #shape(N,), the i-th component contains the log density for sample xi
    return marginal_logdens

def compute_joint_log_densities(X, gmm):
    N_samples = X.shape[1]
    M_components= len(gmm)
    S = numpy.zeros((M_components, N_samples))

    for j in range (M_components):
        mean=gmm[j][1]
        covariance=gmm[j][2]
        S[j,:] = logpdf_GAU_ND(X, mean, covariance)
        weight=gmm[j][0]
        S[j, :] += numpy.log(weight)
    return S

def compute_EM_algorithm(X, initial_GMM, threshold = 10**(-6), constrained=False , psi=-1, covariance_type="full"):
    # Expectation-Maximization algorithm use to find the parameters that best fit the data
    # gmm = [(w1,mu1, C1), (w2,mu2,C2),...]
    estimated_gmm = initial_GMM.copy()
    first = True

    while(True):
        # Expectation STEP - compute the responsibilities: probability that each data point belongs to each component
        # matrix S (num_gmm,num_data)
        joint_log_densities = compute_joint_log_densities(X, estimated_gmm) 
        marginal_log_densities = logpdf_GMM(X, estimated_gmm) 
        log_posterior_distribution= joint_log_densities - marginal_log_densities
        # responsabilities (1 for each component and for each sample = num_gmm x num_samples)
        posterior_distributions = numpy.exp(log_posterior_distribution) 
        
        avg_log_likelihood_nuova = numpy.average(marginal_log_densities)
        
        if (first):
            first=False
        else:
            if (avg_log_likelihood_nuova < avg_log_likelihood_vecchia):
                print("Attenzione, errore")
            if (avg_log_likelihood_nuova - avg_log_likelihood_vecchia < threshold):
                #print("average log likelihood:", avg_log_likelihood_vecchia)
                break
        
        avg_log_likelihood_vecchia = avg_log_likelihood_nuova
        
        # Maximization step: update parameters 

        Zg = posterior_distributions.sum(axis=1) #1 number for each of the GMM components
        Fg = []
        for i in range (posterior_distributions.shape[0]): #1 for each component
            Fg.append((posterior_distributions[i:i+1, :]*X).sum(axis=1))
        
        Sg= []
        for i in range (posterior_distributions.shape[0]):
            Sg.append(0) 
            for j in range (X.shape[1]):
                Sg[i]+=posterior_distributions[i][j]*numpy.dot(X[:, j:j+1],X[:, j:j+1].T )
        
        tied_cov=0
        for i in range (len(estimated_gmm)):
            newM = mcol(Fg[i] / Zg[i])
            newS = Sg[i] / Zg[i] - numpy.dot(newM, newM.T) #4x4 (4= dimensione sample)
            
            if (covariance_type=="Diagonal" or covariance_type=="Tied Diagonal"): 
                newS = newS*numpy.eye(newS.shape[0]) #only the diagonal
            
            if (covariance_type=="Tied" or covariance_type=="Tied Diagonal") : 
                tied_cov += newS*Zg[i]
            
            if ((covariance_type!="Tied" or covariance_type=="Tied Diagonal") and constrained): newS = newCov_constrained(newS, psi)
            
            newW = Zg[i] / Zg.sum()
            estimated_gmm[i] = (newW, newM, newS)
        
        if (covariance_type=="Tied" or covariance_type=="Tied Diagonal"):
            tied_cov=tied_cov/X.shape[1]
            if (constrained): tied_cov=newCov_constrained(tied_cov, psi)
            for i in range (len(estimated_gmm)):
                estimated_gmm[i]= (estimated_gmm[i][0], estimated_gmm[i][1], tied_cov)


    return (estimated_gmm)

def compute_LBG_algorithm(X, number_components, constrained=False , psi=-1, covariance_type="Full", alpha=0.1, delta_l=10**(-6)):

    x = X
    mean = x.mean(1).reshape((x.shape[0], 1))

    X_centered = x - mean
    cov =  numpy.dot(X_centered, X_centered.T)/ x.shape[1]
    
    if (covariance_type =="Diagonal" or covariance_type =="Tied Diagonal"): cov=cov*numpy.eye(cov.shape[0])
    if (constrained): cov=newCov_constrained(cov, psi)
    
    initial_gmm = [(1.0, mean, cov )]
    
    estimated_GMM = initial_gmm
    while (len(estimated_GMM)<number_components):
        estimated_GMM = split2GMM(estimated_GMM, alpha)
        estimated_GMM = compute_EM_algorithm(x, initial_GMM= estimated_GMM, threshold=delta_l, constrained=constrained,psi= psi, covariance_type=covariance_type)
    return estimated_GMM

def split2GMM(gmm, alpha= 0.1):
    splittedGmm = []
    for component in gmm:
        w = component[0]
        mu = component[1]
        sigma = component[2]
        U, s, Vh = numpy.linalg.svd(sigma)
        d = U[:,0:1] * s[0]**0.5 * alpha
        newComp1= (w/2, mu+d, sigma)
        newComp2= (w/2, mu-d, sigma)

        splittedGmm.append(newComp1)
        splittedGmm.append(newComp2)
        
    return splittedGmm

def newCov_constrained (cov, psi):
    U,s,_ =numpy.linalg.svd(cov)
    s[s<psi]=psi
    covNew= numpy.dot(U, mcol(s) * U.T)
    return covNew

def GMM_computeLogLikelihoodRatios(DTR, LTR, DTE, otherParams): 
    constrained= otherParams[0]
    psi=otherParams[1]
    covariance_type=otherParams[2]
    alpha=otherParams[3]
    number_components=otherParams[4]
    delta_l=otherParams[5]

    #train a gmm for each class
    n_classes = len(numpy.unique(LTR))
    gmm_classes = []
    log_class_conditional_distribution= numpy.zeros((n_classes, DTE.shape[1]))
    #obtain one gmm for each class
    for i in range (n_classes):
        data = DTR[:, LTR == i]
        gmm_classes.append(compute_LBG_algorithm(data, number_components= number_components, constrained=constrained, psi=psi, covariance_type= covariance_type,alpha= alpha, delta_l=delta_l))
        log_class_conditional_distribution[i,:]=mrow(logpdf_GMM(GMM=gmm_classes[i], X= DTE))
     
    llr = log_class_conditional_distribution[1,:] - log_class_conditional_distribution[0,:]
    return llr #loglikelihoodRatio
    

