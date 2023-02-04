import numpy
import matplotlib.pyplot as plt
from numpy.random import permutation

def mrow(v):
    return v.reshape((1, v.size))

def mcol(v):
    return v.reshape((v.size, 1))

def means(samples, labels): 
    means = []

    for i in range (numpy.unique(labels).size):
        class_samples =samples[:, labels==i] 
        means.append(class_samples.mean(1))  
    return means 

def covariances(samples, labels, means):
    covariances= []

    for i in range (numpy.unique(labels).size): 
        class_samples=samples[:, labels==i]
        # compute the covariance matrix
        # center the data removing the mean from all points
        centered_samples=class_samples - mcol(means[i]) 
        covariance_matrix=numpy.dot(centered_samples, centered_samples.T) / centered_samples.shape[1]
        covariances.append(covariance_matrix)

    return covariances

def compute_confusion_matrix(predicted_labels, actual_labels, numClasses):
    #Build confusion matrix 
    c_matrix_C =numpy.zeros((numClasses,numClasses))
    #columns =classes, #rows= predictions

    # classLabels: evaluation labels -> actual class labels
    # predicted_labelsC ->assigned class Labels    
    for i in range (len(actual_labels)):
        columnIndex=actual_labels[i]
        rowIndex=predicted_labels[i]
        c_matrix_C[rowIndex][columnIndex]+=1
    return c_matrix_C

def compute_optimal_bayes_decision(loglikelihood_ratios, prior, cost_fn, cost_fp):
    # The optimal decision is the one that minimizes the expected cost
    threshold = - numpy.log((prior*cost_fn)/((1-prior)*cost_fp))
    optimal_decision = (1*(loglikelihood_ratios>threshold))
    return optimal_decision

def compute_emp_pi_T(LTR):
    # empirical prior for true class (high value wines)
    N = LTR.size #tot number of samples
    n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
    pi_emp_T = n_T / N

    return (pi_emp_T)

def compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp):
    FNR = conf_matrix[0][1] /(conf_matrix[0][1] + conf_matrix[1][1] )
    FPR = conf_matrix[1][0] /(conf_matrix[0][0] + conf_matrix[1][0] )

    bayes_risk = prior*cost_fn*FNR+(1-prior)*cost_fp*FPR
    return bayes_risk

def compute_normalized_bayes_risk(bayes_risk ,prior, cost_fn, cost_fp):
    return bayes_risk/(min(prior*cost_fn, (1-prior)*cost_fp))

def compute_loglikelihood(sample, mu, sigma): 
    # Compute the log likelihood of a given sample
    M = sample.shape[0] 
    a = (-M/2) * numpy.log(2*numpy.pi)
    b = (-0.5) * numpy.log( numpy.linalg.det(sigma) )
    
    norm = sample-mu
    sigma_inv = numpy.linalg.inv(sigma)
    
    c=numpy.dot(sigma_inv, norm)
    c = -0.5 *numpy.dot(norm.T, c)
    res = a+b+c
    return res

def compute_loglikelihoods(samples, means, covariances, numlabels):
    #score result matrix
    S= numpy.zeros((numlabels, samples.shape[1])) 
    
    #for each sample compute the likelihood for every class
    for nClass in range (numlabels):
        for  j in range (samples.shape[1]):
            sample = samples[:, j]
            mean =means[nClass]
            covariance =covariances[nClass]
            loglikelihood = compute_loglikelihood(sample, mean, covariance)
            S[nClass][j] = loglikelihood
    return S

def compute_minimum_detection_cost(llrs, labels, prior, cost_fn, cost_fp):

    # Sort log likelihood ratios (scores)
    llrs_sorted= numpy.sort(llrs)
    # treat each llr as threshold and obtain predicted labels by comparing values with it 
    DCFs=[]
    FPRs=[]
    TPRs=[]

    for t in llrs_sorted:
        # compare the score with thresold value and provide the predicted label
        p_label=1*(llrs>t)
        # compute the confusion matrix
        conf_matrix=compute_confusion_matrix(p_label, labels, numpy.unique(labels).size )
        # compute Bayes risk, which is the expected cost of misclassification under the prior probabilities of the classes
        bayes_risk = compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp)
        # Bayes risk divided by the cost of a misclassification. 
        norm_bayes_risk = compute_normalized_bayes_risk(bayes_risk, prior, cost_fn, cost_fp)
        DCFs.append(norm_bayes_risk)

        # Save FPR e TPR
        # FPR 
        FPRs.append(conf_matrix[1][0] /(conf_matrix[0][0] + conf_matrix[1][0] ))
        # 1- FNR
        TPRs.append(1-(conf_matrix[0][1] /(conf_matrix[0][1] + conf_matrix[1][1] )))
    
    DCF_min =min(DCFs)

    index_t = DCFs.index(DCF_min)
    
    return (DCF_min, FPRs, TPRs, llrs_sorted[index_t])

def compute_actual_DCF(llrs, labels, prior , cost_fn, cost_fp):
    # predicted labels using the theoretical threshold
    p_label=compute_optimal_bayes_decision(llrs, prior, cost_fn, cost_fp)
    # build confusion matrix
    conf_matrix=compute_confusion_matrix(p_label, labels, numpy.unique(labels).size )
    # bayes risk
    bayes_risk= compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp)
    # normalized bayes risk -> actDCF
    norm_bayes_risk= compute_normalized_bayes_risk(bayes_risk, prior, cost_fn, cost_fp)
   
    return (norm_bayes_risk)

def k_cross_loglikelihoods(D,L, k, llr_calculator, otherParams):
    # perform k-fold cross-validation for log likelihood ratios
    # Split the data into k folds
    step = int(D.shape[1]/k)
    numpy.random.seed(seed=0)

    random_indexes = permutation(D.shape[1])

    llr = []
    labels = []

    # one fold used for evaluation, the others for training
    for i in range(k):
        if i == k-1:
            indexesEV = random_indexes[i*step:]
            indexesTR = random_indexes[:i*step]
            
        elif i==0:
            indexesEV = random_indexes[0:step]
            indexesTR = random_indexes[step:]

        else:
            indexesEV = random_indexes[i*step:(i+1)*step]
            tmp1 = random_indexes[: i*step]
            tmp2 = random_indexes[(i+1)*step:]
            indexesTR = numpy.concatenate((tmp1,tmp2), axis=None)

        DTR = D[:, indexesTR]
        LTR = L[indexesTR]

        DEV = D[:, indexesEV]
        LEV = L[indexesEV]
        
        llr_i= llr_calculator(DTR, LTR, DEV, otherParams)
        llr.append(llr_i)
        labels.append(LEV)

    llr = numpy.concatenate(llr)
    labels = numpy.concatenate(labels)
    return (llr, labels)

def k_cross_DCF(D, L, k, llr_calculator, prior, cost_fn, cost_fp, otherParams=None, eval_data=None): 
    if (eval_data != None):
        DTE = eval_data[0]
        labels = eval_data[1]
        llr = llr_calculator(D, L, DTE, otherParams)
    else:
        llr, labels = k_cross_loglikelihoods(D, L, k, llr_calculator, otherParams)

    min_DCF, _, _, optimal_threshold = compute_minimum_detection_cost(llr, labels, prior, cost_fn, cost_fp)
    act_DCF = compute_actual_DCF(llr, labels, prior, cost_fn, cost_fp)
    
    return (min_DCF, act_DCF, optimal_threshold)

def bayes_error_plot(D, L, k, llr_calculator, otherParams, title, color, eval_data=None ):
    if (eval_data!=None): 
        DEV = eval_data[0]
        labels= eval_data[1]
        llr = llr_calculator(D, L, DEV, otherParams)
    else :
        llr, labels = k_cross_loglikelihoods(D,L,k, llr_calculator, otherParams)


    effPriorLogOdds = numpy.linspace(-2,2,20)
    effPriors = 1 / (1+ numpy.exp(-effPriorLogOdds))
    dcf = []
    mindcf = []

    for effPrior in effPriors:
        #calculate actual dcf considering effPrior
        d = compute_actual_DCF(llr, labels, effPrior , 1, 1)
        #calculate min dcf considering effPrior
        min_DCF,_,_,_ =compute_minimum_detection_cost(llr, labels, effPrior , 1, 1)
        dcf.append(d)
        mindcf.append(min_DCF)
    
    plt.plot(effPriorLogOdds, dcf, color ,label=title+' DCF')
    plt.plot(effPriorLogOdds, mindcf, color+"--", label=title+ ' min DCF')
    plt.ylim([0,1.1])
    plt.xlim([-2,2])
    plt.legend()

def DCF_optimal_threshold(D, L, k, llr_calculator, otherParams, prior, cost_fn, cost_fp ):
                
    # Calculate the loglikelihood ratios using k-cross method
    llr, labels = k_cross_loglikelihoods(D, L, k, llr_calculator, otherParams)
    
    # Shuffle the scores and then split the shuffled scores into 2 partitions
    num_scores = llr.size
    perm = permutation(num_scores)
    ## shuffle llrs and labels
    llr = llr[perm]
    labels = labels[perm]
    ## split into 2 partitions
    llr1 = llr[:int(num_scores/2)]
    llr2 = llr[int(num_scores/2):]
    labels1 = labels[: int(num_scores/2)]
    labels2 = labels[int(num_scores/2):]

    # Estimate the threshold on one partition, apply the threshold on the other partition
    # and compare minimum and actual DCFs over the latter
    
    # minDCF 
    minDCF,_,_,optimal_treshold = compute_minimum_detection_cost(llr1, labels1, prior , cost_fn, cost_fp)

    predicted_labels = 1*(llr2 > optimal_treshold)

    conf_matrix=compute_confusion_matrix(predicted_labels, labels2, numpy.unique(labels2).size )
    bayes_risk= compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp)

    #norm_bayes_risk is the DCF obtained with the estimated optimal treshold
    norm_bayes_risk = compute_normalized_bayes_risk(bayes_risk, prior, cost_fn, cost_fp)
    
    # actual DCF done with theoretical optimal treshold
    actDCF = compute_actual_DCF(llr2, labels2, prior , cost_fn, cost_fp)

    #minDCF 
    minDCF,_,_,_ = compute_minimum_detection_cost(llr2, labels2, prior , cost_fn, cost_fp)


    return (minDCF, actDCF, norm_bayes_risk, optimal_treshold)