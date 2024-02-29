import h5py
import sklearn.model_selection

def load_dataset(dataset, key):
    with h5py.File(dataset, 'r') as file:
        dats = file[key]
        datss = dats[()]
    return datss.reshape(len(datss), 99)


def create_xtrain_xtest():
    datt = load_dataset('NuGun_preprocessed.h5', 'full_data_cyl')
    x_train ,x_test = sklearn.model_selection.train_test_split(datt)
    return x_train, x_test