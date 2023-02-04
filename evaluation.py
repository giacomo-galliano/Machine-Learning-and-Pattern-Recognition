import numpy
import pre_processing
import utils
import dim_red_techniques
import matplotlib.pyplot as plt
import MVG
import LR
import SVM
import GMM

def table_MVG_eval(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data):
    def MVG_Classifiers(data, eval_data):
        #Full_Cov 
        min_DCF_MVG,_,_ = utils.k_cross_DCF(data, LTR, k, MVG.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("MVG: ",min_DCF_MVG)  

        #Diag_Cov == Naive
        min_DCF_Diag_Cov,_,_ = utils.k_cross_DCF(data, LTR,k, MVG.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("MVG with Diag cov: ",min_DCF_Diag_Cov)

        #Tied
        min_DCF_Tied,_,_ = utils.k_cross_DCF(data, LTR,k, MVG.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("Tied MVG: ",min_DCF_Tied)

        #Tied Diag_Cov
        min_DCF_Tied_Diag_Cov,_,_ = utils.k_cross_DCF(data, LTR, k,  MVG.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[",k," Folds] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)

        print()

    DTE = eval_data[0]
    LTE = eval_data[1]

    normalized_data = pre_processing.z_normalization(DTR)
    normalized_data_eval = pre_processing.z_normalization(DTE)
    pca10 = dim_red_techniques.PCA(normalized_data, 10)
    pca9 = dim_red_techniques.PCA(normalized_data, 9)
    pca10_eval = dim_red_techniques.PCA(normalized_data_eval, 10)
    pca9_eval = dim_red_techniques.PCA(normalized_data_eval, 9)

    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    gaussianizedFeatures_eval = pre_processing.gaussianization(DTE)
    gaussianized_pca_10 = pre_processing.gaussianization(pca10)
    gaussianized_pca_10_eval = pre_processing.gaussianization(pca10_eval)
    gaussianized_pca_9 = pre_processing.gaussianization(pca9)
    gaussianized_pca_9_eval = pre_processing.gaussianization(pca9_eval)

    ######################## NOMRALIZED RAW FEATURES ########################
    print("\n##### minDCF - RAW (normalized) FEATURES - NO PCA #####")
    MVG_Classifiers(normalized_data, [normalized_data_eval, LTE])
    
    ######################## NOMRALIZED RAW FEATURES WITH PCA = 10 ########################
    print("\n##### minDCF - RAW (normalized) FEATURES -  PCA (m=10) #####")
    MVG_Classifiers(pca10, [pca10_eval, LTE])       

    ######################## NOMRALIZED RAW FEATURES WITH PCA = 9 ########################
    
    print("\n##### minDCF - RAW (normalized) FEATURES -  PCA (m=9) #####")
    MVG_Classifiers(pca9, [pca9_eval, LTE])    

    # Z normalization --> PCA dim_reduction --> gaussianization
    ######################## GAUSSIANIZED FEATURES ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES - NO PCA #####")
    MVG_Classifiers(gaussianizedFeatures, [gaussianizedFeatures_eval, LTE])

    ######################## GAUSSIANIZED FEATURES WITH PCA = 10 ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES -  PCA m=10 #####")
    MVG_Classifiers(gaussianized_pca_10, [gaussianized_pca_10_eval, LTE])     

    ######################## GAUSSIANIZED FEATURES WITH PCA = 9 ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES -  PCA m=9 #####")
    MVG_Classifiers(gaussianized_pca_9, [gaussianized_pca_9_eval, LTE])

def table_LR(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data):

    def LR_minDCF(data, eval_data):
            
        lam = 10**(-3)
        pi_T = 0.5
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("lam = 10^-3, pi_T = 0.5: \n- minDCF: ",min_DCF_LR)  

        pi_T = 0.1
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T],eval_data=eval_data)
        print("lam = 10^-3, pi_T = 0.1: \n- minDCF: ",min_DCF_LR)

        pi_T = 0.9
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("lam = 10^-3, pi_T = 0.9: \n- minDCF: ",min_DCF_LR)

    DTE = eval_data[0]
    LTE = eval_data[1]

    normalized_data = pre_processing.z_normalization(DTR)  
    normalized_data_eval = pre_processing.z_normalization(DTE)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    gaussianizedFeatures_eval = pre_processing.gaussianization(DTE)

    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF - RAW (normalized) FEATURES #####")
    LR_minDCF(normalized_data, [normalized_data_eval, LTE])

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES #####")
    LR_minDCF(gaussianizedFeatures, [gaussianizedFeatures_eval, LTE])

def table_Quadratic_LR(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data):

    def Quad_LR_minDCF(data, eval_data):
        lam = 10**(-3)
        
        pi_T = 0.5
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("lam = 10^-3, pi_T = 0.5: \n- minDCF: ",min_DCF_LR)  

        pi_T = 0.1
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("lam = 10^-3, pi_T = 0.1: \n- minDCF: ",min_DCF_LR)

        pi_T = 0.9
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("lam = 10^-3, pi_T = 0.9: \n- minDCF: ",min_DCF_LR)
        
    DTE = eval_data[0]
    LTE = eval_data[1]

    normalized_data = pre_processing.z_normalization(DTR)  
    normalized_data_eval = pre_processing.z_normalization(DTE)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    gaussianizedFeatures_eval = pre_processing.gaussianization(DTE)

    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF - RAW (normalized) FEATURES #####")
    Quad_LR_minDCF(normalized_data, [normalized_data_eval, LTE])

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES #####")
    Quad_LR_minDCF(gaussianizedFeatures, [gaussianizedFeatures_eval, LTE])

def table_SVM_linear(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data):

    def linear_SVM_minDCF(data, eval_data):
        
        C = 0.1
        pi_T = 0.5
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C], eval_data=eval_data)
        print("C= 0.1, pi_T=0.5: \n- minDCF: ",min_DCF)  

        C = 0.1
        pi_T = 0.1
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C], eval_data=eval_data)
        print("C= 0.1, pi_T=0.1: \n- minDCF: ",min_DCF)

        C = 0.1
        pi_T = 0.9
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C], eval_data=eval_data)
        print("C= 0.1, pi_T=0.9: \n- minDCF: ",min_DCF)

    DTE = eval_data[0]
    LTE = eval_data[1]

    normalized_data = pre_processing.z_normalization(DTR)  
    normalized_data_eval = pre_processing.z_normalization(DTE)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    gaussianizedFeatures_eval = pre_processing.gaussianization(DTE)

    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF - RAW (normalized) FEATURES #####")
    linear_SVM_minDCF(normalized_data, [normalized_data_eval, LTE])

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES #####")
    linear_SVM_minDCF(gaussianizedFeatures, [gaussianizedFeatures_eval, LTE])

def table_SVM_quadratic(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data): 

    def quadratic_SVM_minDCF(data, C, c, K, eval_data):
        
        pi_T = 0.5
        min_DCF,act_DCF,_ = utils.k_cross_DCF(data, LTR, k, SVM.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K], eval_data=eval_data)
        print("C=", C, " , pi_T=0.5, c= ", c, " k = ",K ," \n- minDCF: ",min_DCF,"\n- actDCF: ", act_DCF)  
        
        pi_T = 0.1
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K], eval_data=eval_data)
        print("C=", C, " , pi_T=0.1, c= ", c, " k = ",K ," \n- minDCF: ",min_DCF)

        pi_T = 0.9
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K], eval_data=eval_data)
        print("C=", C, " , pi_T=0.9, c= ", c, " k = ",K, " \n- minDCF: ",min_DCF)

        pi_T = utils.compute_emp_pi_T(LTR)
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR, k, SVM.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp,[pi_T, C, c, K], eval_data=eval_data)
        print("C=", C, " , pi_T=pi_emp_T, c= ", c, " k = ",K , " \n- minDCF: ",min_DCF)

    DTE = eval_data[0]
    LTE = eval_data[1]

    C = 0.1
    c = 1
    K = 0

    normalized_data = pre_processing.z_normalization(DTR)  
    normalized_data_eval = pre_processing.z_normalization(DTE)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    gaussianizedFeatures_eval = pre_processing.gaussianization(DTE)
        
    
    print("\nPARAMETRI: (C = " + str(C) + " c= "+ str(c) + "K= " + str(K)+ ")") 

    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF - RAW (normalized) FEATURES #####")
    quadratic_SVM_minDCF(normalized_data,C=C, c=c, K=K, eval_data=[normalized_data_eval, LTE])

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES #####")
    quadratic_SVM_minDCF(gaussianizedFeatures,C=C, c=c, K=K, eval_data=[gaussianizedFeatures_eval, LTE])
    
def table_SVM_RBF(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data): 

    def RBF_SVM_minDCF(data, C, loglam, eval_data):
        
        pi_T = 0.5
        min_DCF,act_DCF,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam], eval_data=eval_data)
        print("C= ", C, ", loglam= ", loglam, " pi_T=0.5: \n- minDCF: ",min_DCF,"\n- actDCF: ", act_DCF)   
        
        pi_T = 0.1
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam], eval_data=eval_data)
        print("C= ", C, ", loglam= ", loglam, " pi_T=0.1: \n- minDCF: ",min_DCF)   
        
        pi_T = 0.9
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam], eval_data=eval_data)
        print("C= ", C, ", loglam= ", loglam, " pi_T=0.9: \n- minDCF: ",min_DCF)   

        # empirical prior
        pi_T = utils.compute_emp_pi_T(LTR)
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam], eval_data=eval_data)
        print("C= ", C, ", loglam= ", loglam, " pi_T=emp_pi_T: \n- minDCF: ",min_DCF)   

    DTE = eval_data[0]
    LTE = eval_data[1]

    C=10**0.1
    loglam=-0.5

    normalized_data = pre_processing.z_normalization(DTR)  
    normalized_data_eval = pre_processing.z_normalization(DTE)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    gaussianizedFeatures_eval = pre_processing.gaussianization(DTE)
        
    print("\nPARAMETRI: (C = " + str(C) + " loglam= "+ str(loglam)+ ")") 

    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF - RAW (normalized) FEATURES #####")
    RBF_SVM_minDCF(normalized_data,C=C, loglam=loglam, eval_data=[normalized_data_eval, LTE])

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES #####")
    RBF_SVM_minDCF(gaussianizedFeatures,C=C, loglam=loglam, eval_data=[gaussianizedFeatures_eval, LTE])

def table_GMM(DTR, LTR, k, eval_data):
    def plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, title):
        widthbar = 0.2

        x_ind = numpy.arange(len(gmm_comp))

        raw_ind = x_ind - widthbar/2
        gau_ind = x_ind + widthbar/2

        lb1 = "minDCF (prior=0.5) - Raw"
        lb2 = "minDCF (prior=0.5) - Gaussianized"
        
        plt.figure()
        plt.bar(raw_ind, raw_minDCFs, width = widthbar, color = 'orange', label = lb1)
        plt.bar(gau_ind, gau_minDCFs, width = widthbar, color = 'green', label = lb2)
        plt.title(title)
        plt.xticks(x_ind ,gmm_comp)
        plt.ylabel('minDCFs')
        plt.xlabel('GMM components')
        plt.legend(loc="lower left")

        plt.savefig('Plots/Results/GMM_'+title+'.png' )

    def GMM_compute_DCFs(DTR, LTR, k, covariance_type, prior, cost_fn, cost_fp, eval_data):

        DTE = eval_data[0]
        LTE = eval_data[1]

        gmm_comp = [1,2,4,8,16,32]

        raw_minDCFs = []
        gau_minDCFs = []

        normalized_data = pre_processing.z_normalization(DTR)
        gaussianizedFeatures = pre_processing.gaussianization(DTR)

        constrained=True
        psi=0.01
        alpha=0.1
        delta_l=10**(-6)
    
        for i in range(len(gmm_comp)):
            params = [constrained, psi, covariance_type, alpha, gmm_comp[i],delta_l]
            print("-------> working on raw data, component= ", gmm_comp[i])
            # Raw features
            raw_minDCFs_i,_,_ = utils.k_cross_DCF(normalized_data, LTR, k, GMM.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params, [pre_processing.z_normalization(DTE), LTE])
            print("RAW DATA, num components = " + str(gmm_comp[i]) + ", minDCF = " + str(raw_minDCFs_i) )
            # Gaussianized features
            print("-------> working on gauss data, component= ", gmm_comp[i])
            gau_minDCFs_i,_,_ = utils.k_cross_DCF(gaussianizedFeatures, LTR,k, GMM.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params, [pre_processing.gaussianization(DTE), LTE])
            print("GAUSS DATA, num components = " + str(gmm_comp[i]) + ", minDCF = " + str(gau_minDCFs_i) )
            raw_minDCFs.append(raw_minDCFs_i)
            gau_minDCFs.append(gau_minDCFs_i)
            print()    
        
        raw_minDCFs=numpy.array(raw_minDCFs)
        gau_minDCFs=numpy.array(gau_minDCFs)
        return raw_minDCFs, gau_minDCFs, gmm_comp


    #### Full Cov
    covariance_type = "Full"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1, eval_data)
    plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Full_covariance")

    #### Diagonal Cov
    covariance_type = "Diagonal"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1, eval_data)
    plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Diagonal_covariance")


    #### Diagonal Cov
    covariance_type = "Tied"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1, eval_data)
    plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Tied_covariance")

    
    #### Diagonal Cov
    covariance_type = "Tied Diagonal"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1, eval_data)
    plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Tied_Diagonal_covariance")

def bayes_error_plots(data, L, k, llr_calculators, other_params, titles, colors, eval_data):
    plt.figure()
    plt.title("Bayes Error Plot")
    plt.xlabel("prior log odds")
    plt.ylabel("DCF")
    for i in range (len(llr_calculators)):
        print("Plotting Bayes error calculator in position "+ str(i))
        utils.bayes_error_plot(data[i], L, k, llr_calculators[i], other_params[i], titles[i], colors[i], eval_data)

    plt.savefig('Plots/Results/Error_Bayes_Plot_eval.png' )

