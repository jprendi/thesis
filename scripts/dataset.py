import h5py
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from qkeras import quantized_bits
from qkeras.utils import _add_supported_quantized_objects
from qkeras import quantized_bits

def load_dataset(dataset, key='full_data_cyl', pid=False):
    """
    Load a dataset from an HDF5 file.

    Parameters:
        dataset (str): The path to the HDF5 file.
        key (str): The key to access the dataset within the HDF5 file.

    Returns:
        numpy.ndarray: The loaded dataset.
    """
    if dataset == 'BSM' or dataset == 'sig':
        dataset = 'BSM_preprocessed.h5'

    if dataset == 'bkg' or dataset == 'NuGun':
        dataset == 'NuGun_preprocessed.h5'

    with h5py.File(dataset, 'r') as file:
        dats = file[key]
        datss = dats[()]

    if pid==True:
        data = datss.reshape(len(datss), 99)
        nulls = get_zeros(data)
        return remove_columns(data, nulls)
    else:
        return datss.reshape(len(datss), 99)


def create_xtrain_xtest(random_seed=1, background_data='NuGun_preprocessed.h5', bkg_key='full_data_cyl', pid=False):
    """
    Create training and testing datasets from background data.

    Parameters:
        random_seed (int): Random seed for reproducibility.
        background_data (str): The path to the background data HDF5 file.
        bkg_key (str): The key to access the background data within the HDF5 file.

    Returns:
        tuple: A tuple containing train and test datasets.
    """
    datt = load_dataset(background_data, bkg_key, pid)
    x_train ,x_test = train_test_split(datt, random_state=random_seed)
    return x_train, x_test



def supervised_xtrain_xtest(sig_key, signal_data='BSM_preprocessed.h5', random_seed=1, background_data='NuGun_preprocessed.h5', bkg_key='full_data_cyl'):
    """
    Create supervised training and testing datasets by combining background and signal data.

    Parameters:
        sig_key (str): The key to access the signal data within the signal data HDF5 file.
        signal_data (str): The path to the signal data HDF5 file.
        random_seed (int): Random seed for reproducibility.
        background_data (str): The path to the background data HDF5 file.
        bkg_key (str): The key to access the background data within the HDF5 file.

    Returns:
        tuple: A tuple containing training and testing datasets along with their corresponding labels.
    """
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


def create_CAE_datset(test_size=0.2, val_size=0.2):
    """
    Create train, test and validation set from the background and prepares it in a way that can be used as a direct input for the CAE defined in the CAE.py script.
    Parameters:
        test_size (int): fixes how big the test size is
        val_size (str): fixes how big the val size is
    Returns:
        tuple: A tuple containing train, validation and test datasets in the right shape for the CAE.
    """
        
    with h5py.File('NuGun_preprocessed.h5', 'r') as file:
        dats = file['full_data_cyl']
        datss = dats[()]
        np.random.seed(1) 
        np.random.shuffle(datss)

    X_train, X_test = train_test_split(datss, test_size=test_size, shuffle=True)
    X_train, X_val = train_test_split(X_train, test_size=val_size)
    return np.reshape(X_train, (-1, 33,3,1)), np.reshape(X_test, (-1, 33,3,1)), np.reshape(X_val, (-1, 33,3,1))


def load_sig_CAE(key):
    """
    Create train, test and validation set from the background and prepares it in a way that can be used as a direct input for the CAE defined in the CAE.py script.
    Parameters:
        key (str): key that indicates which BSM signal is to be loaded
    Returns:
        numpy.ndarray: a numpy ndarray of the loaded signal datasets in the right shape for the CAE to use for prediction
    """
    with h5py.File('BSM_preprocessed.h5', 'r') as file:
        dats = file[key]
        datss = dats[()]
    return np.reshape(datss, (-1, 33,3,1))


def load_axo_dataset(dataset, key='full_data_cyl'):
    columns_to_remove = [98, 97, 96, 95, 94, 93, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]
    X_flat = remove_columns(load_dataset(dataset, key), columns_to_remove)

    input_quantizer = quantized_bits(8,5,alpha=1)
    scales = h5py.File('scales.h5')
    scale_data = scales['norm_scale'][:].flatten()
    offset_data = scales['norm_bias'][:].flatten()
    return input_quantizer((X_flat.astype('float') - offset_data) / scale_data)


def get_all_bsm_keys(dataset='BSM_preprocessed.h5'):
    """
    Get all keys from a BSM dataset.

    Parameters:
        dataset (str): The path to the BSM dataset HDF5 file.

    Returns:
        list: A list of tuples containing keys and their corresponding shapes.
    """
    listing = []
    with h5py.File(dataset, 'r') as file:
        for i in file.keys():
            listing.append([i, np.shape(file[i])])
    return listing


def bsm_keys(dataset='BSM_preprocessed.h5'):
    """
    Get BSM keys having a shape of (3,).

    Parameters:
        dataset (str): The path to the BSM dataset HDF5 file.

    Returns:
        list: A list of tuples containing keys and their corresponding shapes.
    """
    lists = get_all_bsm_keys(dataset)
    potential=[]
    for i in range(len(lists)):
        if np.shape(lists[i][1]) == (3,):
            potential.append(lists[i])
    return potential

def get_zeros():
    '''
    Returns a list of the zero-entried columns
    '''
    # # transposed_dataset = zip(*dataset)
    # # zero_columns = []
    # # for i, column in enumerate(transposed_dataset):
    # #     if all(element == 0 for element in column):
    # #         zero_columns.append(i)
    # # return zero_columns[::-1]  
        #   this is not efficient and not consistent. so i just return the list that i know is true every time here !
    return [62, 61, 60, 59, 58, 57, 1]


def remove_columns(dataset, list):
    '''
    Removes given columns of dataset.
    '''
    for column in list:
        dataset = np.delete(dataset, column, axis=1)
    return dataset


def inject_signal(bkg, sig, size='max', percentage=0.01, random_seed=1):
    """
    Inject signal into background data.

    Parameters:
        bkg (numpy.ndarray): Background data.
        sig (numpy.ndarray): Signal data.
        size (int/str): Number of samples in the new dataset. If 'max', the size will be the maximum of background and background + percentage of background.
        percentage (float): Percentage of background data to be included in the new dataset.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the new dataset, labels, and details of injected signal.
    """
    random.seed(random_seed)

    dim1 = np.shape(bkg)
    dim2 = np.shape(sig)
    
    if size == 'max':
        size = int(dim1[0] + dim1[0] * percentage)
    else:
        size = size

    size1 = int((1 - percentage) * size)
    size2 = int(percentage * size)

    size1 = int((1 - percentage) * size)
    size2 = int(percentage * size)

    if size2 > dim2[0]:
        size = dim2[0]/percentage
        size1 = int((1 - percentage) * size)
        size2 = int(percentage * size)
        n2 = random.sample(range(dim2[0]), size2)
        n1 = random.sample(range(dim1[0]), size1)
    else:
        n1 = random.sample(range(dim1[0]), size1)
        n2 = random.sample(range(dim2[0]), size2)

    newdat1 = bkg[n1]
    newdat2 = sig[n2]
    labels = np.concatenate((np.zeros(size1), np.ones(size2)))
    new_dataset = np.concatenate((newdat1, newdat2))

    return new_dataset, labels, (newdat1, newdat2), (size1, size2)
