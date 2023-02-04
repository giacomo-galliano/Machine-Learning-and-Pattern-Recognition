import numpy
import load_data
import plot_functions
import pre_processing
import validation 
import evaluation
import SVM

if __name__ == '__main__':

    ## DTR: Training Data  
    ## DTE: Evaluation Data
    ## LTR: Training Labels
    ## LTE: Evaluation Labels

    DTR, LTR = load_data.load_data('data/Train.txt')
    DTE, LTE = load_data.load_data('data/Test.txt')

    ###########################################
    ############ FEATURES ANALYSIS ############
    ###########################################

    ## Training set - composition and plots
    high_quality_samples = numpy.count_nonzero(LTR == 1)
    low_quality_samples = numpy.count_nonzero(LTR == 0)
    print('Training set:\n- High quality samples: ', high_quality_samples, '\n-Low quality samples: ', low_quality_samples)


    ## Test set - composition and plots
    high_quality_samples = numpy.count_nonzero(LTE == 1)
    low_quality_samples = numpy.count_nonzero(LTE == 0)
    print('Test set:\n- High quality samples: ', high_quality_samples, '\n-Low quality samples: ', low_quality_samples)

    if(False):
        ## Training set - RAW features plot
        plot_functions.histogram(DTR, LTR, 'Plots/Analysis/Raw/Histograms', 'raw')

        ## Normalization
        z_norm = pre_processing.z_normalization(DTR)
        plot_functions.histogram(z_norm, LTR, 'Plots/Analysis/Normalized/Histograms', 'z_norm')

        ## Gaussianization
        gauss = pre_processing.gaussianization(DTR)
        plot_functions.histogram(gauss, LTR, 'Plots/Analysis/Gaussianized/Histograms', 'gauss')

        # Look for correlation between features 
        plot_functions.heat_map(DTR, LTR, 'Plots/Analysis/Raw/HeatMaps')
        plot_functions.heat_map(z_norm, LTR, 'Plots/Analysis/Normalized/HeatMaps')
        plot_functions.heat_map(gauss, LTR, 'Plots/Analysis/Gaussianized/HeatMaps')

    ###########################################
    ############ VALIDATION PHASE #############
    ###########################################
    print('\n\n###########################################\n############ VALIDATION PHASE #############\n###########################################\n\n')
    k = 5

    if(False):
        # MVG classifiers
        print('\n####### MVG CLASSIFIERS #######\n')
        print("\n---- pi = 0.5\n")
        validation.table_MVG(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k)
        print()
        print("\n---- pi = 0.1\n")
        validation.table_MVG(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k)
        print()
        print("\n---- pi = 0.9\n")
        validation.table_MVG(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k)
        print()

        # Logistic Regression
        ## Linear Logistic Regression
        print('\n####### LINEAR LOGISTIC REGRESSION CLASSIFIERS #######\n')
        print("####### LR GRAPHS lambda tuning #######")
        validation.graphs_LR_lambdas(DTR,LTR, k=k)
        
        print("\n####### LR TABLE #######")
        print("\n- applicazione prior = 0.5")
        validation.table_LR(DTR,LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k)
        print("\n- applicazione con prior = 0.1")
        validation.table_LR(DTR,LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k)
        print("\n- applicazione con prior = 0.9")
        validation.table_LR(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k)

        ## Quadratic Logistic Regression
        print('\n####### QUADRATIC LOGISTIC REGRESSION CLASSIFIERS #######\n')
        print("####### QUADRATIC LR GRAPHS lambda tuning #######")
        validation.graphs_quadratic_LR_lambdas(DTR, LTR,  k)
        
        print("\n####### QUADRATIC LR TABLE #######")
        print("\n- applicazione prior = 0.5")
        validation.table_Quadratic_LR(DTR,LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k)
        print()
        print("\n- applicazione con prior = 0.1")
        validation.table_Quadratic_LR(DTR,LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k)
        print()
        print("\n- applicazione con prior = 0.9")
        validation.table_Quadratic_LR(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k)

        # SVM - Support Vector Machines
        ## Linear SVM
        print('\n####### LINEAR SVM CLASSIFIERS #######\n')
        print("####### SVM Linear GRAPHS #######")
        validation.graphs_SVM_Cs(DTR, LTR, k=k )
        
        print("\n####### SVM LINEAR TABLE #######")
        print("\n- applicazione con prior = 0.5")
        validation.table_SVM_linear(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
        print("\n- applicazione con prior = 0.9")
        validation.table_SVM_linear(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
        print("\n- applicazione con prior = 0.1")
        validation.table_SVM_linear(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )

        ## SVM w/quadratic kernel
        print('\n####### SVM W/QUADRATIC KERNEL CLASSIFIERS #######\n')
        print("####### SVM Quadratic GRAPHS changing C,c,k #######")
        validation.graphs_Polinomial_SVM_Cs_k_c(DTR, LTR, k=k )
        print("####### SVM Quadratic GRAPHS #######")
        validation.graphs_Polinomial_SVM_Cs(DTR, LTR, k=k )

        print("\n####### SVM QUADRATIC TABLE #######")
        print("\n- applicazione con prior = 0.5")
        validation.table_SVM_quadratic(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
        print("\n- applicazione con prior = 0.9")
        validation.table_SVM_quadratic(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
        print("\n- applicazione con prior = 0.1")
        validation.table_SVM_quadratic(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )

    if(False):
        ## SVM w/RBF kernel
        print('\n####### SVM W/RBF KERNEL CLASSIFIERS #######\n')
        print("####### SVM RBF GRAPHS #######")
        validation.graphs_RBF_SVM_Cs(DTR, LTR, k=k )

        print("\n####### SVM RBF TABLE #######")
        print("\n- applicazione con prior = 0.5")
        validation.table_SVM_RBF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
        print("\n- applicazione con prior = 0.9")
        validation.table_SVM_RBF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
        print("\n- applicazione con prior = 0.1")
        validation.table_SVM_RBF(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )

        # GMM _ Gaussian Mixture Model
        print('\n####### GAUSSIAN MIXTURE MODEL CLASSIFIERS #######\n')
        validation.graphs_GMM(DTR, LTR, k)

        #### ACTUAL DCF VS MIN DCF ####
        # We chose the two best performing models and we compared the results in terms of minDCF and actDCF.
        # The chosen models are SVM w/quadratic kernel and the SVM w/RBF kernel.
        # Now we need to understand if the classifiers are already well calibrated or not; to do so, the best way is to use
        # a Bayes error plot.

    # BAYES ERROR PLOT
    ## SVM Quadratic kernel
    C_q = 0.1
    c_q = 1
    K_q = 0
    pi_T_q = 0.5
    
    ## SVM RBF kernel
    C_rbf = 10**0.1
    lam_rbf = 10**-0.5
    pi_T_rbf = 0.5

    norm_data = pre_processing.z_normalization(DTR)
    data = [norm_data, norm_data]
    llr_calculators = [SVM.SVM_computeLogLikelihoods, SVM.RBF_SVM_computeLogLikelihoods]
    params = [[pi_T_q, C_q, c_q, K_q], [pi_T_rbf, C_rbf, lam_rbf]]
    titles = ['SVM Quad', 'SVM RBF']
    colors = ['r', 'b']
    # print('\n\n########### BAYES ERROR PLOT ###########\n\n')
    # validation.bayes_error_plots(data, LTR, k, llr_calculators, params, titles, colors )

        # CLOSE-TO-OPTIMAL THRESHOLD ESTIMATION
        # To address the miscalibration problem we estimate an application dependent threshold (close-to-optimal threshold)
        # How? We consider computing the minDCF for that particular application, and then we can just take the corresponding threshold of the min DCF on the validation set.
        # This will not produce well-calibrated scores for a different number of applications; if we change application, we'll need to perform again the process.

    print('\n\n########### CLOSE-TO-OPTIMAL THRESHOLD ESTIMATION ###########\n\n')
    print("\n- applicazione con prior = 0.5")
    validation.treshold_estimated_table(data[0], LTR, 0.5, 1, 1, k, llr_calculators[0], params[0], titles[0])
    validation.treshold_estimated_table(data[1], LTR, 0.5, 1, 1, k, llr_calculators[1], params[1], titles[1])

    print("\n- applicazione con prior = 0.1")
    validation.treshold_estimated_table(data[0], LTR, 0.1, 1, 1, k, llr_calculators[0], params[0], titles[0])
    validation.treshold_estimated_table(data[1], LTR, 0.1, 1, 1, k, llr_calculators[1], params[1], titles[1])
    
    print("\n- applicazione con prior = 0.9")
    validation.treshold_estimated_table(data[0], LTR, 0.9, 1, 1, k, llr_calculators[0], params[0], titles[0])
    validation.treshold_estimated_table(data[1], LTR, 0.9, 1, 1, k, llr_calculators[1], params[1], titles[1])

##############################################################################################################

    ###########################################
    ############ EVALUATION PHASE #############
    ###########################################
    print('\n\n###########################################\n############ EVALUATION PHASE #############\n###########################################\n\n')

    eval_data = [DTE, LTE]

    print("############ only pi = 0.5 ############")

    if(False):
        # MVG classifiers
        print('\n####### MVG CLASSIFIERS #######\n')
        evaluation.table_MVG_eval(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)

        # Logistic Regression
        ## Linear Logistic Regression
        print('\n####### LINEAR LOGISTIC REGRESSION CLASSIFIERS #######\n')
        evaluation.table_LR(DTR,LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)

        ## Quadratic Logistic Regression
        print('\n####### QUADRATIC LOGISTIC REGRESSION CLASSIFIERS #######\n')
        evaluation.table_Quadratic_LR(DTR,LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)

        # SVM - Support Vector Machines
        ## Linear SVM
        print('\n####### LINEAR SVM CLASSIFIERS #######\n')
        evaluation.table_SVM_linear(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)

        ## SVM w/quadratic kernel
        print('\n####### SVM W/QUADRATIC KERNEL CLASSIFIERS #######\n')
        evaluation.table_SVM_quadratic(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)

        ## SVM w/RBF kernel
        print('\n####### SVM W/RBF KERNEL CLASSIFIERS #######\n')
        evaluation.table_SVM_RBF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)

    if(False):
        # GMM _ Gaussian Mixture Model
        print('\n####### GAUSSIAN MIXTURE MODEL CLASSIFIERS #######\n')
        evaluation.table_GMM(DTR, LTR, k, eval_data = eval_data)

        #### ACTUAL DCF VS MIN DCF ####
        # We chose the two best performing models and we compared the results in terms of minDCF and actDCF.
        # The chosen models are SVM w/quadratic kernel and the SVM w/RBF kernel.
        # Now we need to understand if the classifiers are already well calibrated or not; to do so, the best way is to use
        # a Bayes error plot.

        # BAYES ERROR PLOT
        ## SVM Quadratic kernel
        C_q = 0.1
        c_q = 1
        K_q = 0
        pi_T_q = 0.5
        
        ## SVM RBF kernel
        C_rbf = 10**0.1
        lam_rbf = 10**-0.5
        pi_T_rbf = 0.5

        norm_data = pre_processing.z_normalization(DTR)
        norm_data_eval = pre_processing.z_normalization(DTE)
        data = [norm_data, norm_data]
        llr_calculators = [SVM.SVM_computeLogLikelihoods, SVM.RBF_SVM_computeLogLikelihoods]
        params = [[pi_T_q, C_q, c_q, K_q], [pi_T_rbf, C_rbf, lam_rbf]]
        titles = ['SVM Quad', 'SVM RBF']
        colors = ['r', 'b']
        print('\n\n########### BAYES ERROR PLOT ###########\n\n')
        evaluation.bayes_error_plots(data, LTR, k, llr_calculators, params, titles, colors, eval_data=[norm_data_eval, LTE])
