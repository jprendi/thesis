import h5py
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

def load_dataset(dataset, key):
    with h5py.File(dataset, 'r') as file:
        dats = file[key]
        datss = dats[()]
    return datss.reshape(len(datss), 99)


def create_xtrain_xtest(random_seed=1, background_data='NuGun_preprocessed.h5', bkg_key='full_data_cyl'):
    datt = load_dataset(background_data, bkg_key)
    x_train ,x_test = train_test_split(datt, random_state=random_seed)
    return x_train, x_test




def supervised_xtrain_xtest(sig_key, signal_data = 'BSM_preprocessed.h5', random_seed=1, background_data='NuGun_preprocessed.h5', bkg_key='full_data_cyl'):
    le = LabelEncoder()
    bkg = load_dataset(background_data, bkg_key)
    sig = load_dataset(signal_data, sig_key)
    x_train_bkg, x_test_bkg = train_test_split(bkg, random_state=random_seed)
    x_train_sig, x_test_sig = train_test_split(sig, random_state=random_seed)

    x_train = shuffle(np.concatenate((x_train_bkg, x_train_sig)), random_state=random_seed)
    y_train = shuffle(to_categorical(le.fit_transform(np.concatenate((np.zeros(len(x_train_bkg)), np.ones(len(x_train_sig)))))), random_state=random_seed)

    x_test = shuffle(np.concatenate((x_test_bkg, x_test_sig)), random_state=random_seed)
    y_test = shuffle(to_categorical(le.fit_transform(np.concatenate((np.zeros(len(x_test_bkg)), np.ones(len(x_test_sig)))))), random_state=random_seed)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test



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
