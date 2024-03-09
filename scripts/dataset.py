import h5py
import random
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(dataset, key):
    with h5py.File(dataset, 'r') as file:
        dats = file[key]
        datss = dats[()]
    return datss.reshape(len(datss), 99)


def create_xtrain_xtest(random_seed=1, background_data='NuGun_preprocessed.h5', bkg_key='full_data_cyl'):
    datt = load_dataset(background_data, bkg_key)
    x_train ,x_test = train_test_split(datt, random_state=random_seed)
    return x_train, x_test


def inject_signal(bkg, sig, size, percentage, random_seed=1):
    
    random.seed(random_seed)

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

    return new_dataset, labels, (newdat1, newdat2), (size1, size2)



def get_all_bsm_keys(dataset='BSM_preprocessed.h5'):
    listing = []
    with h5py.File(dataset, 'r') as file:
        for i in file.keys():
            listing.append([i, np.shape(file[i])])
    return listing


def bsm_keys(dataset='BSM_preprocessed.h5'):
    lists = get_all_bsm_keys(dataset)
    potential=[]
    for i in range(len(lists)):
        if np.shape(lists[i][1]) == (3,):
            potential.append(lists[i])
    return potential
