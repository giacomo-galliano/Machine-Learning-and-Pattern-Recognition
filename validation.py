import numpy
import matplotlib.pyplot as plt
import pre_processing
import utils
import dim_red_techniques 
import MVG
import LR
import SVM
import GMM

def table_MVG(DTR, LTR, prior, cost_fn, cost_fp, k):

    def MVG_Classifiers(data):
        #Full_Cov 
        min_DCF_MVG, act_DCF_MVG,_ = utils.k_cross_DCF(data, LTR, k, MVG.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("MVG Full Cov: \n- minDCF:",min_DCF_MVG,"\n- actDCF: ", act_DCF_MVG)  

        #Diag_Cov (Naive)
        min_DCF_Diag_Cov,act_DCF_Diag_Cov,_ = utils.k_cross_DCF(data, LTR,k, MVG.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("MVG Diag Cov: \n- minDCF:",min_DCF_Diag_Cov,"\n- actDCF: ", act_DCF_Diag_Cov)  

        #Tied
        min_DCF_Tied,act_DCF_Tied,_ = utils.k_cross_DCF(data, LTR,k, MVG.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("MVG Tied: \n- minDCF:",min_DCF_Tied,"\n- actDCF: ", act_DCF_Tied)  

        #Tied Diag_Cov
        min_DCF_Tied_Diag_Cov,act_DCF_Tied_Diag_Cov,_ = utils.k_cross_DCF(data, LTR, k,  MVG.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("MVG Tied Cov: \n- minDCF:",act_DCF_Tied_Diag_Cov,"\n- actDCF: ", act_DCF_Tied_Diag_Cov)  

    normalized_data = pre_processing.z_normalization(DTR)  
    pca10 = dim_red_techniques.PCA(normalized_data, 10)
    pca9 = dim_red_techniques.PCA(normalized_data, 9)
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    gaussianized_pca_10 = pre_processing.gaussianization(pca10)
    gaussianized_pca_9 = pre_processing.gaussianization(pca9)

    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF & actDCF - RAW (normalized) FEATURES - NO PCA #####")
    MVG_Classifiers(normalized_data)

    ######################## NORMALIZED RAW FEATURES WITH PCA = 10 ########################
    print("\n##### minDCF & actDCF - RAW (normalized) FEATURES -  PCA (m=10) #####")
    MVG_Classifiers(pca10)       

    ######################## NORMALIZED RAW FEATURES WITH PCA = 9 ########################
    print("\n##### minDCF & actDCF - RAW (normalized) FEATURES -  PCA (m=9) #####")
    MVG_Classifiers(pca9)       


    # Z normalization --> PCA dim_reduction --> gaussianization
    ######################## GAUSSIANIZED FEATURES ########################
    print("\n##### minDCF & actDCF - GAUSSIANIZED FEATURES - NO PCA #####")
    MVG_Classifiers(gaussianizedFeatures)

    ######################## GAUSSIANIZED FEATURES WITH PCA = 10 ########################
    print("\n##### minDCF & actDCF - GAUSSIANIZED FEATURES -  PCA m=10 #####")
    MVG_Classifiers(gaussianized_pca_10)     

    ######################## GAUSSIANIZED FEATURES WITH PCA = 9 ########################
    print("\n##### minDCF & actDCF - GAUSSIANIZED FEATURES -  PCA m=9 #####")
    MVG_Classifiers(gaussianized_pca_9)     

def graphs_LR_lambdas(DTR, LTR,  k):

    def graph_kfold(data, prior, cost_fn, cost_fp, pi_T):
        print("Printing graph kfold prior = ", prior)
        exps = numpy.linspace(-8,5, 14)
        lambdas = 10** exps
        minDCFs = 0 * exps
        k=5
        for i in range (lambdas.size):
            lam= lambdas[i]
            minDCFs[i],_,_ = utils.k_cross_DCF(data, LTR,k, LR.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()

    normalized_data = pre_processing.z_normalization(DTR)
    gaussianizedFeatures = pre_processing.gaussianization(DTR)

    plt.figure()
    print("\nRaw features, ",k," fold")
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(normalized_data, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(normalized_data, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Plots/Validation/LR_linear_5FoldRAW.png' )

    plt.figure()
    print("\nGaussianized features, ",k," fold")
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Plots/Validation/LR_linear_5FoldGauss.png' )

def table_LR(DTR, LTR, prior, cost_fn, cost_fp, k):

    def LR_minDCF(data):
        # lambda value obtained after tuning in previous step
        lam = 10**(-3)
        # lam = 10**(-2)

        pi_T = 0.5
        min_DCF_LR,act_DCF_LR,_ = utils.k_cross_DCF(data, LTR, k, LR.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("lam = 10^-3, pi_T = 0.5: \n- minDCF: ",min_DCF_LR,"\nact_DCF: ", act_DCF_LR)  

        pi_T = 0.1
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("lam = 10^-3, pi_T = 0.1: \n- minDCF: ",min_DCF_LR)

        pi_T = 0.9
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("lam = 10^-3, pi_T = 0.9: \n- minDCF: ",min_DCF_LR)

        # empirical prior
        pi_T = utils.compute_emp_pi_T(LTR)
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("lam = 10^-3, pi_T = emp_pi_t: \n- minDCF: ",min_DCF_LR)

    normalized_data = pre_processing.z_normalization(DTR)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    
    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF & actDCF - RAW (normalized) FEATURES #####")
    LR_minDCF(normalized_data)

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF & actDCF - GAUSSIANIZED FEATURES #####")
    LR_minDCF(gaussianizedFeatures)

def graphs_quadratic_LR_lambdas(DTR, LTR,  k):
    def graph_kfold(data, prior, cost_fn, cost_fp, pi_T):
        print("Printing graph kfold prior = ", prior)
        exps = numpy.linspace(-8,5, 14)
        lambdas = 10** exps
        minDCFs = 0 * exps
        k=5
        for i in range (lambdas.size):
            lam= lambdas[i]
            minDCFs[i],_,_ = utils.k_cross_DCF(data, LTR,k, LR.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()

    normalized_data = pre_processing.z_normalization(DTR)
    gaussianizedFeatures = pre_processing.gaussianization(DTR)

    plt.figure()
    print("Raw features, 5 fold")
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(normalized_data, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(normalized_data, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Plots/Validation/LR_quadratic_5FoldRAW.png' )

    plt.figure()
    print("Gaussianized features, 5 fold")
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Plots/Validation/LR_quadratic_5FoldGauss.png' )

def table_Quadratic_LR(DTR, LTR, prior, cost_fn, cost_fp, k):

    def Quad_LR_minDCF(data):
        # lambda value obtained after tuning in previous step
        # lam = 10**(-3)
        lam = 10**(-3)
        
        pi_T = 0.5
        min_DCF_LR,act_DCF_LR,_ = utils.k_cross_DCF(data, LTR, k, LR.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("lam = 10^-3, pi_T = 0.5: \n- minDCF: ",min_DCF_LR,"\nact_DCF: ", act_DCF_LR)  

        pi_T = 0.1
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("lam = 10^-3, pi_T = 0.1: \n- minDCF: ",min_DCF_LR)

        pi_T = 0.9
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("lam = 10^-3, pi_T = 0.9: \n- minDCF: ",min_DCF_LR)

        # empirical prior
        pi_T = utils.compute_emp_pi_T(LTR)
        min_DCF_LR,_,_ = utils.k_cross_DCF(data, LTR, k, LR.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("lam = 10^-3, pi_T = emp_pi_t: \n- minDCF: ",min_DCF_LR)

    normalized_data = pre_processing.z_normalization(DTR)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    
    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF & actDCF - RAW (normalized) FEATURES #####")
    Quad_LR_minDCF(normalized_data)

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF & actDCF - GAUSSIANIZED FEATURES #####")
    Quad_LR_minDCF(gaussianizedFeatures)

def graphs_SVM_Cs(DTR, LTR, k ):
    def graph_kfold(data, prior, cost_fn, cost_fp, pi_T):
        print("Printing graph kfold prior = ", prior)
        exps = numpy.linspace(-3,1, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
        lb = "minDCF (prior="+ str(prior) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        

    normalized_data = pre_processing.z_normalization(DTR)
    gaussianizedFeatures = pre_processing.gaussianization(DTR)

    # empirical prior 
    pi_emp_T = utils.compute_emp_pi_T(LTR)

    print("Plotting Raw features, 5 fold, without class balancing")
    plt.figure()
    plt.title("Raw features, 5 fold, without class balancing")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T = pi_emp_T)
    graph_kfold(normalized_data, prior=0.1, cost_fn=1, cost_fp=1, pi_T = pi_emp_T)
    graph_kfold(normalized_data, prior=0.9, cost_fn=1, cost_fp=1, pi_T = pi_emp_T)
    plt.savefig('Plots/Validation/SVM_linear_5FoldNOClassBalance.png' )

    print("Plotting Raw features, 5 fold, with class balancing")
    plt.figure()
    plt.title("Raw features, 5 fold, with class balancing")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    graph_kfold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Plots/Validation/SVM_linear_5FoldClassBalance.png' )

def table_SVM_linear(DTR, LTR, prior, cost_fn, cost_fp, k):

    def linear_SVM_minDCF(data):
        C = 0.1
        
        pi_T = 0.5
        min_DCF,act_DCF,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        print("C= 0.1, pi_T=",pi_T,": \n- minDCF: ",min_DCF,"\nact_DCF: ", act_DCF)  

        pi_T = 0.1
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        print("C= 0.1, pi_T=",pi_T,": \n- minDCF: ",min_DCF)
        
        pi_T = 0.9
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        print("C= 0.1, pi_T=",pi_T,": \n- minDCF: ",min_DCF)

        # empirical prior
        pi_T = utils.compute_emp_pi_T(LTR)
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        print("C= 0.1, pi_T=emp_pi_T: \n- minDCF: ",min_DCF)

    normalized_data = pre_processing.z_normalization(DTR)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    
    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF & actDCF - RAW (normalized) FEATURES #####")
    linear_SVM_minDCF(normalized_data)

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF & actDCF - GAUSSIANIZED FEATURES #####")
    linear_SVM_minDCF(gaussianizedFeatures)

def graphs_Polinomial_SVM_Cs(DTR, LTR, k ):

    def graph_kfold(data, prior, cost_fn, cost_fp, pi_T, K, c):
        print("Printing graph kfold prior = ",prior)
        exps = numpy.linspace(-2,2, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = utils.k_cross_DCF(data, LTR,k, SVM.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        
        lb = "minDCF (prior= ", prior, ")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        

    normalized_data = pre_processing.z_normalization(DTR)
    gaussianizedFeatures = pre_processing.gaussianization(DTR)

    plt.figure()
    print("Plotting Raw features, 5 fold")
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    graph_kfold(normalized_data, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    graph_kfold(normalized_data, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5,  K=0.0, c=1.0)
    plt.savefig('Plots/Validation/SVM_quadratic_5FoldRAW.png' )
    
    plt.figure()
    print("Plotting Gaussianized features, 5 fold")
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    graph_kfold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    graph_kfold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5,  K=0.0, c=1.0)
    
    plt.savefig('Plots/Validation/SVM_quadratic_5FoldGAU.png' )

def graphs_Polinomial_SVM_Cs_k_c(DTR, LTR, k ):

    def graph_kfold(data, prior, cost_fn, cost_fp, pi_T, K, c):
        print("Printing graph kfold \nK = ", K, "c = ", c)
        exps = numpy.linspace(-2,2, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = utils.k_cross_DCF(data, LTR,k, SVM.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        
        lb = "minDCF (k="+ str(K) +" c= "+ str(c)+ ")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        

    normalized_data = pre_processing.z_normalization(DTR)
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    
    plt.figure()
    print("Plotting Raw features, 5 fold")
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=0.0)
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=0.0)
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=1.0)

    plt.savefig('Plots/Validation/SVM_quadratic_kc_5FoldRAW_kc.png' )
    
    plt.figure()
    print("Plotting Gaussianized features, 5 fold")
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=0.0)
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=0.0)
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=1.0)

    plt.savefig('Plots/Validation/SVM_quadratic_kc_5FoldGAU_kc.png' )

def table_SVM_quadratic(DTR, LTR, prior, cost_fn, cost_fp, k): 

    def quadratic_SVM_minDCF(data, C, c, K):
        
        pi_T = 0.5
        min_DCF,act_DCF,_ = utils.k_cross_DCF(data, LTR, k, SVM.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("C= ", C, ", pi_T=",pi_T,", c= ", c, " k = ",K ,"  \n- minDCF: ",min_DCF,"\nact_DCF: ", act_DCF)  
        
        pi_T = 0.1
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("C= ", C, ", pi_T=",pi_T,", c= ", c, " k = ",K ," \n- minDCF: ", min_DCF)
        
        pi_T = 0.9
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("C=", C, ", pi_T=",pi_T,", c= ", c, " k = ",K, " \n- minDCF: ", min_DCF)
        
        # empirical prior
        pi_T = utils.compute_emp_pi_T(LTR)
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("C=", C, ", pi_T=emp_pi_T, c= ", c, " k = ",K, " \n- minDCF: ", min_DCF)

    # C=100, c=1, k=0
    # C=0.1, c=1, k=1
    C = 0.1
    c = 1
    K = 0

    normalized_data = pre_processing.z_normalization(DTR)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    
    print("\nPARAMETRI: (C =", C, "\tc = ", c,"\tK = ", K,")") 

    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF - RAW (normalized) FEATURES #####")
    quadratic_SVM_minDCF(normalized_data,C=C, c=c, K=K)

    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES #####")
    quadratic_SVM_minDCF(gaussianizedFeatures,C=C, c=c, K=K)

def graphs_RBF_SVM_Cs(DTR, LTR, k):

    def graph_kfold(data, prior, cost_fn, cost_fp, pi_T, loglam):
        print("Plotting k fold loglam = ", loglam)
        exps = numpy.linspace(-1,2, 10)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        
        lb = " (log(lam)="+ str(loglam) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        

    normalized_data = pre_processing.z_normalization(DTR)
    gaussianizedFeatures = pre_processing.gaussianization(DTR)

    print("Plotting Raw features, 5 fold")
    plt.figure()
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0)
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0.5)
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0.8)
    graph_kfold(normalized_data, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 1)
    plt.savefig('Plots/Validation/SVM_RBF_5FoldRAW.png' )

    print("Plotting Gaussinized features, 5 fold")
    plt.figure()
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0)
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0.5)
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0.8)
    graph_kfold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 1)
    plt.savefig('Plots/Validation/SVM_RBF_5FoldGauss.png' )

def table_SVM_RBF(DTR, LTR, prior, cost_fn, cost_fp, k): 

    def RBF_SVM_minDCF(data, C, loglam):
        
        pi_T = 0.5
        min_DCF,act_DCF,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("C= ", C, ", loglam= ", loglam, " pi_T=",pi_T,": \n- minDCF: ",min_DCF, "\n- actDCF: ", act_DCF)   
        
        pi_T = 0.1
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("C= ", C, ", loglam= ", loglam, " pi_T=",pi_T,": \n- minDCF: ",min_DCF)   

        pi_T = 0.9
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("C= ", C, ", loglam= ", loglam, " pi_T=",pi_T,": \n- minDCF: ",min_DCF)   

        # empirical prior
        pi_T = utils.compute_emp_pi_T(LTR)
        min_DCF,_,_ = utils.k_cross_DCF(data, LTR,k, SVM.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("C= ", C, ", loglam= ", loglam, " pi_T=emp_pi_T: \n- minDCF: ",min_DCF)   

    C=10**0.1 
    loglam=-0.5

    normalized_data = pre_processing.z_normalization(DTR)  
    gaussianizedFeatures = pre_processing.gaussianization(DTR)
    
    print("\nPARAMETRI: (C = ", C,"\tloglam = ",loglam,")") 

    ######################## NORMALIZED RAW FEATURES ########################
    print("\n##### minDCF - RAW (normalized) FEATURES #####")
    RBF_SVM_minDCF(normalized_data, C=C, loglam=loglam)
    
    ######################## GAUSSINIZED FEATURES ########################
    print("\n##### minDCF - GAUSSIANIZED FEATURES #####")
    RBF_SVM_minDCF(gaussianizedFeatures ,C=C, loglam=loglam)

def graphs_GMM(DTR, LTR, k):

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

        plt.savefig('Plots/Validation/'+title+'.png' )


    def GMM_compute_DCFs(DTR, LTR, k, covariance_type, prior, cost_fn, cost_fp):
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
            raw_minDCFs_i,_,_ = utils.k_cross_DCF(normalized_data, LTR, k, GMM.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params)
            print("RAW DATA, num components = " + str(gmm_comp[i]) + ", minDCF = " + str(raw_minDCFs_i) )
            # Gaussianized features
            print("-------> working on gauss data, component= ", gmm_comp[i])
            gau_minDCFs_i,_,_ = utils.k_cross_DCF(gaussianizedFeatures, LTR,k, GMM.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params)
            print("GAUSSIANIZED DATA, num components = " + str(gmm_comp[i]) + ", minDCF = " + str(gau_minDCFs_i) )
            raw_minDCFs.append(raw_minDCFs_i)
            gau_minDCFs.append(gau_minDCFs_i)
            print()    
        
        raw_minDCFs=numpy.array(raw_minDCFs)
        gau_minDCFs=numpy.array(gau_minDCFs)
        return raw_minDCFs, gau_minDCFs, gmm_comp

    
    #### Full Cov
    covariance_type = "Full"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1)
    plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Full_covariance")

    #### Diagonal Cov
    covariance_type = "Diagonal"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1)
    plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Diagonal_covariance")

    #### Diagonal Cov
    covariance_type = "Tied"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1)
    plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Tied_covariance")
    
    #### Diagonal Cov
    covariance_type = "Tied Diagonal"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1)
    plot_histogram_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Tied_Diagonal_covariance")

def bayes_error_plots(data, L, k, llr_calculators, other_params, titles, colors):
    plt.figure()
    plt.title("Bayes Error Plot")
    plt.xlabel("prior log odds")
    plt.ylabel("DCF")
    for i in range (len(llr_calculators)):
        print("Plotting Bayes error calculator in position "+ str(i))
        utils.bayes_error_plot(data[i], L, k, llr_calculators[i], other_params[i], titles[i], colors[i] )

    plt.savefig('Plots/Validation/Error_Bayes_Plot_val.png' )

def treshold_estimated_table(data, LTR, prior, cost_fn, cost_fp, k, llr_calculator, otherParams, title):
    
    _, _, actDCF_opt, _ = utils.DCF_optimal_threshold(data, LTR, k, llr_calculator, otherParams, prior, cost_fn, cost_fp )

    print(title,': close to optimal DCF = ', actDCF_opt)
