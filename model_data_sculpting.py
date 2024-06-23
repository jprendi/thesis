from scripts import physics, dataset   
import pickle 
import numpy as np
import matplotlib.pyplot as plt

class sculpt_study():
    def __init__(self, model_dir, model_name, signal, dimass_type, inv_mass=False):
        self.ibm_color_blind_palette = ["#648fff", "#ffb000", "#785ef0", "#dc267f", "#fe6100", "#8c564b"]
        self.model_dir = model_dir
        self.model_name = model_name
        self.sig_key = signal
        self.dimass_type = dimass_type
        
        self.ZeroBias_data = dataset.load_dataset(dataset = 'L1Ntuple_2023EphZB_run367883_13_1_0_pre4_caloParams_2023_v0_2_preprocessed_sorted.h5')
        _, self.NuGun_data = dataset.create_xtrain_xtest()
        self.Signal_data = dataset.load_dataset(dataset='BSM_preprocessed.h5', key=self.sig_key)
        print('Datasets loaded!')
        if inv_mass == True:
            self.ZeroBias_inv_mass, self.ZeroBias_inv_mass_indices = physics.invariant_mass(self.ZeroBias_data, type=self.dimass_type, return_indices=True)
            self.NuGun_inv_mass, self.NuGun_inv_mass_indices = physics.invariant_mass(self.NuGun_data, type=self.dimass_type, return_indices=True)
            self.Signal_inv_mass, self.Signal_inv_mass_indices = physics.invariant_mass(self.Signal_data, type=self.dimass_type, return_indices=True)
            print(f'Invariant di{self.dimass_type}mass calculated!')

        self.model = pickle.load(open(self.model_dir+self.model_name, 'rb'))
        self.ZeroBias_anomaly_score = self.model.predict(self.ZeroBias_data, output="score")
        self.NuGun_anomaly_score = self.model.predict(self.NuGun_data, output="score")
        self.Signal_anomaly_score = self.model.predict(self.Signal_data, output="score")
        print('Predictions of model loaded!')


    def select_anomalous_datapoints(anomaly_score, dataset, area='top', percentage=0.1):
        if area == 'top':
            sorted_indices = np.argsort(anomaly_score)[::-1]
        elif area == 'bottom':
            sorted_indices = np.argsort(anomaly_score)
        else:
            raise ValueError("indicate area: 'top' or 'bottom' ")    

        percent_elements = int(len(anomaly_score) * percentage)
        selected_datapoints = [dataset[i] for i in sorted_indices[:percent_elements]]
        return np.array(selected_datapoints)
    
    def plot_anomaly_sculpting(self, dataset, anomaly_score, binning=[60, 80, 100, 80, 60], percentage_type=[0.01, 0.1, 0.5, 0.1, 0.01], area_type=['bottom', 'bottom', 'bottom', 'top', 'top'], colors=['gold', 'crimson', 'mediumvioletred','purple', 'midnightblue'], range=None):

        for idx, percentage in enumerate(percentage_type):
            invmass = physics.invariant_mass(self.select_anomalous_datapoints(anomaly_score= anomaly_score, dataset= dataset, area=area_type[idx], percentage=percentage), type='jet') 
            plt.hist(invmass, log=True, histtype='step', color= colors[idx],  bins= binning[idx], label=f'{area_type[idx]} {percentage}', density=True, range=range)
        plt.legend()


    def data_MC_ratio_plots(self, area, percentage, color, binning=100):
        top_1_ZB = np.array(self.select_anomalous_datapoints(self=1, anomaly_score=self.ZeroBias_anomaly_score, dataset=self.ZeroBias_data, area=area, percentage=percentage))
        top_1_NG = np.array(self.select_anomalous_datapoints(self=1, anomaly_score=self.NuGun_anomaly_score, dataset=self.NuGun_data, area=area, percentage=percentage))

        # Create a figure and two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        # Plot something in the first subplot
        NG = ax1.hist(top_1_NG[:,63], density=True, log=True, histtype='step', bins=binning, color=color, label='NuGun')
        ZB = ax1.hist(top_1_ZB[:,63], density=True, log=True, alpha=0.2, bins=binning, color=color, label='ZeroBias')
        ax1.set_xlim(0,2100)

        # Plot something in the second subplot
        ax2.plot( NG[1][:100], ZB[0]/NG[0], '*', color=color)
        ax2.set_xlim(0,2100)
        ax2.set_ylim(0,5)
        ax1.legend()


    def data_MC_ratios(self, area, percentage, color):
        top_1_ZB = physics.invariant_mass(self.select_anomalous_datapoints(self=1, anomaly_score=self.ZeroBias_anomaly_score, dataset=self.ZeroBias_data, area=area, percentage=percentage), type='jet')
        top_1_NG = physics.invariant_mass(self.select_anomalous_datapoints(self=1, anomaly_score=self.NuGun_anomaly_score, dataset=self.NuGun_data, area=area, percentage=percentage), type='jet')
        # Create a figure and two subplots
        # Plot something in the first subplot
        NG = np.histogram(top_1_NG, density=True, bins=100)
        ZB = np.histogram(top_1_ZB, density=True, bins=100)

        # Plot something in the second subplot
        plt.plot( NG[1][:100], NG[0]/ZB[0], '*', color=color)
        plt.show()