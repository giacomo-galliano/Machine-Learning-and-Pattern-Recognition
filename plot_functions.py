import numpy
import matplotlib.pyplot as plt

FEATURES = {
    0: 'fixed acidity',
    1: 'volatile acidity',
    2: 'citric acid',
    3: 'residual sugar',
    4: 'chlorides',
    5: 'free sulfur dioxide',
    6: 'total sulfur dioxide',
    7: 'density',
    8: 'pH',
    9: 'sulphates',
    10: 'alcohol' 
}

def histogram(D, L, path, preprocessing):
    #splt data into high_quality and low_quality
    D_hq = D[:, L==1]
    D_lq = D[:, L==0]

    for feat_index in range(D.shape[0]):
        plt.figure()
        plt.xlabel(FEATURES[feat_index])
        plt.hist(D_lq[feat_index, :], bins = 30, density = True, alpha = 0.4, label = 'Low Quality')
        plt.hist(D_hq[feat_index, :], bins = 30, density = True, alpha = 0.4, label = 'High Quality')
        plt.legend()
        plt.savefig(str(path)+'/histogram_%s_%d.png' % (preprocessing , feat_index))

def heat_map(D, L, path):
    # show features correlation in the whole dataset
    C =numpy.corrcoef(D)
    plt.figure()
    #plt.imshow(C, cmap='Blues')
    plt.imshow(C, cmap='Greys')
    plt.colorbar()
    plt.title("Entire dataset")
    plt.savefig(str(path)+'/entire_dataset')

    # show features correlation in the low quality wine samples
    C =numpy.corrcoef(D[:,L==0])
    plt.figure()
    plt.imshow(C, cmap='Blues')
    #plt.imshow(C, cmap='Oranges')
    plt.colorbar()
    plt.title("Low quality wine samples")
    plt.savefig(str(path)+'/low_quality')

    # show features correlation in the high quality wine samples
    C =numpy.corrcoef(D[:,L==1])
    plt.figure()
    plt.imshow(C, cmap='Reds')
    plt.colorbar()
    plt.title("High quality wine samples")
    plt.savefig(str(path)+'/high_quality')