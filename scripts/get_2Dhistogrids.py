'''
with this script, we do a reweighting of NuGun for it to match ZeroBias better. 
we start with the dijet invariant mass spectrum.
we do individual reweighting of the two different jets by plotting their respective p_T and \eta distribution. 
in both cases we observed a lack in data for the \eta distribution for p_T<30. 
to deal with it, we simply look at data points above the threshold 30.

this function also deals with certain gaps in the 2D distribution of both jets by simply linearly imputing them

'''

from model_data_sculpting import sculpt_study
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


    # here the thing that executes it all....

class create_reweight_maps():
    def __init__(self, sigkey, 
                 model_dir='trained_models/isotree/kfold_models/', 
                 model_name='model__ntrees_100__scoring_metric_depth__fold_1_of_5',
                 dimass_type='jet', grids=False):
        
        """
        ☆★☆ init of the class!

        here the dataset needed for reweighting NuGun is loaded and the weights derived

        input parameteters:
        - sigkey (string): the signal we are probing for
        - model_dir (string): the path to the model
        - model_name (string): the name of the file of the model. otherwise the default is the depth-based iForest
        - dimass_type (string): indicate what type of object we want to look at in terms of dimass spectrum
        - grids (bool): will let us know whether we would like to load the reweighting grid
        """

        self.sculpts = sculpt_study(model_dir=model_dir, model_name=model_name, signal=sigkey, dimass_type=dimass_type, grids=grids)

        self.ZB_Bottom = sculpt_study.select_anomalous_datapoints(self.sculpts.ZeroBias_anomaly_score, self.sculpts.ZeroBias_data, area='bottom', percentage=0.95)
        self.NG_Bottom = sculpt_study.select_anomalous_datapoints(self.sculpts.NuGun_anomaly_score, self.sculpts.NuGun_data, area='bottom', percentage=0.95)

        self.ratio1 = 0
        self.ratio2 = 0
        
        if grids:
            self.get_grids(self.ZB_Bottom, self.NG_Bottom)

            top_ZB_pre = sculpt_study.select_anomalous_datapoints(anomaly_score=self.sculpts.ZeroBias_anomaly_score, dataset=self.sculpts.ZeroBias_data, area='top', percentage=0.05)
            top_NG_pre = sculpt_study.select_anomalous_datapoints(anomaly_score=self.sculpts.NuGun_anomaly_score, dataset=self.sculpts.NuGun_data, area='top', percentage=0.05)
            self.top_NGT = top_NG_pre[(top_NG_pre[:, 63] > 30) & (top_NG_pre[:, 66] > 30)]
            self.NGT_weights = self.get_weights()
            self.top_ZBT = top_ZB_pre[(top_ZB_pre[:, 63] > 30) & (top_ZB_pre[:, 66] > 30)]
        else:
            top_ZB_pre = sculpt_study.select_anomalous_datapoints(anomaly_score=self.sculpts.ZeroBias_anomaly_score, dataset=self.sculpts.ZeroBias_data, area='top', percentage=0.05)
            top_NG_pre = sculpt_study.select_anomalous_datapoints(anomaly_score=self.sculpts.NuGun_anomaly_score, dataset=self.sculpts.NuGun_data, area='top', percentage=0.05)
            self.top_NGT = top_NG_pre
            self.top_ZBT = top_ZB_pre



    def get_weights(self):
        '''
        loads the event by event map that resulted from the other code

        output: list with the event-based weights
        '''
        weights_jet1 = self.dataset_weights(self.top_NGT[:,63], self.top_NGT[:,64], self.ratio1, jet=1)
        weights_jet2 = self.dataset_weights(self.top_NGT[:,63+3], self.top_NGT[:,64+3], self.ratio1, jet=2)
        return list(map(self.multiply_entries, weights_jet1, weights_jet2))

    def multiply_entries(self, entry1, entry2):
        '''
        just a function to multiply entries
        output: the multiplied entries
        '''
        return entry1*entry2

    def get_grids(self, ZB_Bottom, NG_Bottom):
        '''
        creates the grid for both jets based on the p_T and pseudorapidity of the given jet!

        input:
        - ZB_Bottom (numpy.ndarray): the least anomalous data points in ZeroBias
        - NG_Bottom (numpy.ndarray): the least anomalous data points in NuGun 

        '''

        num_bins_x = 60
        quantiles_x = np.linspace(0, 1, num_bins_x + 1)
        bin_edge_x1 = np.quantile(ZB_Bottom[(ZB_Bottom[:, 63] > 30) & (ZB_Bottom[:, 66] > 30), 63], quantiles_x)

        # jet 1
        self.NGB_jet1 = plt.hist2d(NG_Bottom[(NG_Bottom[:, 63] > 30) & (NG_Bottom[:, 66] > 30), 63], 
                NG_Bottom[(NG_Bottom[:, 63] > 30) & (NG_Bottom[:, 66] > 30), 64], 
                bins=[bin_edge_x1,40], norm=mcolors.LogNorm(), cmap='rainbow')
        plt.colorbar(label='Count')
        plt.xlabel('jet 1 p_T')
        plt.ylabel('jet 1 $\\eta$')
        # plt.xlim(30,160)
        plt.show()

        # jet 2
        self.NGB_jet2 =  plt.hist2d(NG_Bottom[(NG_Bottom[:, 63] > 30) & (NG_Bottom[:, 66] > 30), 63+3], 
                NG_Bottom[(NG_Bottom[:, 63] > 30) & (NG_Bottom[:, 66] > 30), 64+3], 
                bins=[bin_edge_x1,40], norm=mcolors.LogNorm(), cmap='rainbow')
        plt.colorbar(label='Count')
        plt.xlabel('jet 2 p_T')
        plt.ylabel('jet 2 $\\eta$')
        plt.show()


        # jet 1
        self.ZBB_jet1 = plt.hist2d(ZB_Bottom[(ZB_Bottom[:, 63] > 30) & (ZB_Bottom[:, 66] > 30), 63], 
                ZB_Bottom[(ZB_Bottom[:, 63] > 30) & (ZB_Bottom[:, 66] > 30), 64], 
                bins=[bin_edge_x1,40], norm=mcolors.LogNorm(), cmap='rainbow')
        plt.colorbar(label='Count')
        plt.xlabel('jet 1 p_T')
        plt.ylabel('jet 1 $\\eta$')
        plt.show()

        # jet 2
        self.ZBB_jet2 =  plt.hist2d(ZB_Bottom[(ZB_Bottom[:, 63] > 30) & (ZB_Bottom[:, 66] > 30), 63+3], 
                ZB_Bottom[(ZB_Bottom[:, 63] > 30) & (ZB_Bottom[:, 66] > 30), 64+3], 
                bins=[bin_edge_x1,40], norm=mcolors.LogNorm(), cmap='rainbow')
        plt.colorbar(label='Count')
        plt.xlabel('jet 2 p_T')
        plt.ylabel('jet 2 $\\eta$')
        plt.show()

        NGB_jet1_new = self.impute_values(self.NGB_jet1)
        NGB_jet2_new = self.impute_values(self.NGB_jet2)
        ZBB_jet1_new = self.impute_values(self.ZBB_jet1)
        ZBB_jet2_new = self.impute_values(self.ZBB_jet2)

        ratio1 = ZBB_jet1_new/NGB_jet1_new
        ratio1[np.isnan(ratio1)] = 0
        ratio1[np.isinf(ratio1)] = 1

        ratio2 = ZBB_jet2_new/NGB_jet2_new
        ratio2[np.isnan(ratio2)] = 0
        ratio2[np.isinf(ratio2)] = 1

        self.ratio1 = ratio1
        self.ratio2 = ratio2



    def search_right_value(self, datapoint_indices_pT, datapoint_indices_eta, grid, max_value=59):
        '''
        finds the value next (to the right) of the given data point. if the right value is empty, it takes the one next to it, etc.

        input:
        - datapoint_indices_pT (int): the index of the data point along p_T
        - datapoint_indices_eta (int): the index of the data point along eta
        - grid (numpy.ndarray): the grid with the weights
        - max_value (int): set to 59 as we have 60 bins along the x-axis

        output:
        - counter (int): this corresponds to the index that is not empty next to the index we started out from

        '''
        counter = datapoint_indices_pT+1
        # print(f"counter before while: {counter}")
        while grid[counter, datapoint_indices_eta] == 0:
            # print(grid[counter, datapoint_indices_eta] == 0)
            # print(f"counter within while {counter}")
            if counter == max_value:
                return max_value
            else:
                # print(f"count {counter}")
                counter += 1
        return counter


    def linear_impute(self, datapoint_indices, grid):
        '''
        this function is used to determine what value one needs to impute and which entries need to be imputed

        input:
        - datapoint_indices (int): the index we start out from (along p_T)
        - grid (numpy.ndarray): the array with the weights

        output:
        - impute_value (float): the value that we impute given empty points with
        - to_impute (list(tuple)): gives the index/indices of the datapoints that need to be imputed

        '''
        datapoint_indices_pT = datapoint_indices[0]
        datapoint_indices_eta = datapoint_indices[1]
        left_value = grid[datapoint_indices_pT-1, datapoint_indices_eta]
        # print(f"pt aentry {datapoint_indices_pT} eta entry {datapoint_indices_eta} in linear impute before search_right_value")
        right_value_index = self.search_right_value(datapoint_indices_pT, datapoint_indices_eta, grid)
        right_value = grid[right_value_index, datapoint_indices_eta]
        impute_value = ((left_value+right_value))/2
        to_impute = list((datapoint_indices_pT, right_value_index-1))
        return impute_value, to_impute


    def impute_values(self, histo):
        '''
        will impute the missing entries from a given 2D histogram

        input: 
        - histo (numpy.ndarray, numpy.ndarray): the result of a given 2D histogram

        output:
        - grid_1 (numpy.ndarray): the 2D grid with the now imputed values 
        
        '''
        grid_1 = histo[0]
        for p_T_index in range(1,len(histo[1])-2):
            for eta_index in range(1,(len(histo[2]))-1):
                if grid_1[p_T_index, eta_index] == 0:
                    # print(f"pt and eta entry is zero: {p_T_index} {eta_index}")
                    impute_value, to_impute = self.linear_impute((p_T_index, eta_index), grid_1)
                    # print(f"impute value {impute_value}")
                    for ind in to_impute:
                        # print(ind)
                        grid_1[ind, eta_index] = impute_value
                        # print(f"imputation {grid_1[ind, eta_index]}, {impute_value}")
        return grid_1


    def find_bin2(self, jet_pt, jet_eta, all_grid):
        '''
        finds the weight of an event for the second jet

        input:
        - jet_pt (float): value of the p_T of jet 2
        - jet_eta (float): value of the eta of jet 2
        - all_grid (numpy.ndarray): the grid of the weights for jet 2

        outut:
        - calibration (float): gives the weight for that event
        '''
        jet_range = self.ZBB_jet2[1]
        eta_range = self.ZBB_jet2[2]

        pT_ind = np.digitize(jet_pt,jet_range[:-1])
        eta_ind = np.digitize(jet_eta,eta_range[:-1])
        calibration = all_grid[pT_ind-1, eta_ind-1]
        return calibration

    def find_bin1(self, jet_pt, jet_eta, all_grid):
        '''
        finds the weight of an event for the first jet

        input:
        - jet_pt (float): value of the p_T of jet 1
        - jet_eta (float): value of the eta of jet 1
        - all_grid (numpy.ndarray): the grid of the weights for jet 1

        outut:
        - calibration (float): gives the weight for that event
        '''
        jet_range = self.ZBB_jet1[1]
        eta_range = self.ZBB_jet1[2]    

        pT_ind = np.digitize(jet_pt,jet_range[:-1])
        eta_ind = np.digitize(jet_eta,eta_range[:-1])
        calibration = all_grid[pT_ind-1, eta_ind-1]
        return calibration
    

    def dataset_weights(self, jet_pts, jet_etas, all_grid, jet):
        '''
        finds the weight of an event for a given jet

        input:
        - jet_pt (float): value of the p_T of a jet
        - jet_eta (float): value of the eta of a jet
        - all_grid (numpy.ndarray): the grid of the weights for the given jet
        - jet(int): to indicate which jet we want to calculate (the first (jet=1) or the second jet (jet=2))
        '''

        if jet == 2:
            return list(self.find_bin2(jet_pts, jet_etas, all_grid))
        else:
            return list(self.find_bin1(jet_pts, jet_etas, all_grid))
