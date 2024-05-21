from scripts import physics, dataset   
import pickle 
import numpy as np
import matplotlib.pyplot as plt

class sculpt_study():
    def __init__(self, model_dir, model_name, signal, dimass_type):
        self.ibm_color_blind_palette = ["#648fff", "#ffb000", "#785ef0", "#dc267f", "#fe6100", "#8c564b"]
        self.model_dir = model_dir
        self.model_name = model_name
        self.sig_key = signal
        self.dimass_type = dimass_type
        
        self.ZeroBias_data = dataset.load_dataset(dataset = 'L1Ntuple_2023EphZB_run367883_13_1_0_pre4_caloParams_2023_v0_2_preprocessed_sorted.h5')
        _, self.NuGun_data = dataset.create_xtrain_xtest()
        self.Signal_data = dataset.load_dataset(dataset='BSM_preprocessed.h5', key=self.sig_key)
        print('Datasets loaded!')

        self.ZeroBias_inv_mass, self.ZeroBias_inv_mass_indices = physics.invariant_mass(self.ZeroBias_data, type=self.dimass_type, return_indices=True)
        self.NuGun_inv_mass, self.NuGun_inv_mass_indices = physics.invariant_mass(self.NuGun_data, type=self.dimass_type, return_indices=True)
        self.Signal_inv_mass, self.Signal_inv_mass_indices = physics.invariant_mass(self.Signal_data, type=self.dimass_type, return_indices=True)
        print(f'Invariant di{self.dimass_type}mass calculated!')

        self.model = pickle.load(open(self.model_dir+self.model_name, 'rb'))
        self.ZeroBias_anomaly_score = self.model.predict(self.ZeroBias_data, output="score")
        self.NuGun_anomaly_score = self.model.predict(self.NuGun_data, output="score")
        self.Signal_anomaly_score = self.model.predict(self.Signal_data, output="score")
        print('Predictions of model loaded!')


    def select_anomalous_datapoints(self, anomaly_score, dataset, area='top', percentage=0.1):
        if area == 'top':
            sorted_indices = np.argsort(anomaly_score)[::-1]
        elif area == 'bottom':
            sorted_indices = np.argsort(anomaly_score)
        else:
            raise ValueError("indicate area: 'top' or 'bottom' ")    

        percent_elements = int(len(anomaly_score) * percentage)
        selected_datapoints = [dataset[i] for i in sorted_indices[:percent_elements]]
        return selected_datapoints
    
    def plot_anomaly_sculpting(self, dataset, anomaly_score, binning=[60, 80, 100, 80, 60], percentage_type=[0.01, 0.1, 0.5, 0.1, 0.01], area_type=['bottom', 'bottom', 'bottom', 'top', 'top'], colors=['gold', 'crimson', 'mediumvioletred','purple', 'midnightblue'], range=None):

        for idx, percentage in enumerate(percentage_type):
            invmass = physics.invariant_mass(self.select_anomalous_datapoints(anomaly_score= anomaly_score, dataset= dataset, area=area_type[idx], percentage=percentage), type='jet') 
            plt.hist(invmass, log=True, histtype='step', color= colors[idx],  bins= binning[idx], label=f'{area_type[idx]} {percentage}', density=True, range=range)
        plt.legend()