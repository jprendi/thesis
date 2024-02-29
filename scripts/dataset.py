import h5py
import sklearn.model_selection
import random
import numpy as np



def load_dataset(dataset, key):
    with h5py.File(dataset, 'r') as file:
        dats = file[key]
        datss = dats[()]
    return datss.reshape(len(datss), 99)


def create_xtrain_xtest():
    datt = load_dataset('NuGun_preprocessed.h5', 'full_data_cyl')
    x_train ,x_test = sklearn.model_selection.train_test_split(datt)
    return x_train, x_test


def inject_signal(bkg, sig, size, percentage):
    dim1 = np.shape(bkg)
    dim2 = np.shape(sig)
    
    if size == 'max':
        size = int(dim1[0] + dim1[0]*percentage)
    else:
        size = size
        ## this part for sure has to be rewritten at some point, depending on the data set :)

    size1 = int((1-percentage)*size)
    size2 = int(percentage*size)

    n1 = random.sample(range(dim1[0]), size1)
    n2 = random.sample(range(dim2[0]), size2)

    newdat1 = bkg[n1]
    newdat2 = sig[n2]
    labels = np.concatenate((np.zeros(size1),np.ones(size2)))
    new_dataset = np.concatenate((newdat1, newdat2))

    return new_dataset, labels, newdat1, newdat2, size1, size2