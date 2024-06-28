'''
this scripts produces the plot that compares the simulated data (NuGun), real data (real_data).
Then with injected signal into real data.
The comparison is with the leading order pT objects. Saves plots!
The MET range is zoomed in. However, there is some peak in all above 4k . :) 
'''

from scripts import dataset
import matplotlib.pyplot as plt

NuGun = dataset.load_dataset(dataset='NuGun_preprocessed.h5')
real_data = dataset.load_dataset(dataset='L1Ntuple_2023EphZB_run367883_13_1_0_pre4_caloParams_2023_v0_2_preprocessed_sorted.h5')


# Create a figure and four subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
MET_range=(0,500)
# Plot for MET
axs[0, 0].hist(NuGun[:,0], log=True, range=MET_range,histtype='step',color=ibm_color_blind_palette[0],bins=100, label='NuGun')
axs[0, 0].hist(real_data[:,0], log=True, range=MET_range, histtype='step', color=ibm_color_blind_palette[1],bins=100, label='real')
axs[0, 0].legend()
axs[0, 0].set_title('MET')

# Plot for p_T e/gamma
axs[0, 1].hist(NuGun[:,3], log=True, histtype='step',color=ibm_color_blind_palette[0], bins=100, label='NuGun')
axs[0, 1].hist(real_data[:,3], log=True, histtype='step',color=ibm_color_blind_palette[1], bins=100, label='real data')
axs[0, 1].legend()
axs[0, 1].set_title('p_T e/gamma')

# Plot for p_T muon
axs[1, 0].hist(NuGun[:,39], log=True, histtype='step',color=ibm_color_blind_palette[0], bins=100, label='NuGun')
axs[1, 0].hist(real_data[:,39], log=True, histtype='step', color=ibm_color_blind_palette[1],bins=100, label='real data')
axs[1, 0].legend()
axs[1, 0].set_title('p_T muon')

# Plot for p_T jet
axs[1, 1].hist(NuGun[:,63], log=True, histtype='step', color=ibm_color_blind_palette[0],bins=100, label='NuGun')
axs[1, 1].hist(real_data[:,63], log=True, histtype='step', color=ibm_color_blind_palette[1],bins=100, label='real data')
axs[1, 1].legend()
axs[1, 1].set_title('p_T jet')

plt.tight_layout()
plt.savefig(f'leading_order_plots/all_nugun_realdata_{MET_range}')
plt.show()





sigkeys = ["ttHto2B", "VBFHToInvisible", "GluGluHToGG_M-90", "HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm"
, "ggXToYYTo2Mu2E_m18", "GluGluHToTauTau", "SMS-Higgsino", "SUSYGluGluToBBHToBB_NarrowWidth_M-120"]

ibm_color_blind_palette = ["#648fff", "#ffb000", "#785ef0", "#dc267f", "#fe6100", "#8c564b"]
MET_range = [(0,1300), (0,1200), (0,500), (0,1300), (0,500), (0,900), (0,1400), (0,850)]

for idx, sig in enumerate(sigkeys):
    signal = dataset.load_dataset(dataset='BSM', key=sig)
    injected_dataset, _,_,_ = dataset.inject_signal(real_data, signal)


    # Create a figure and four subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot for MET
    axs[0, 0].hist(NuGun[:,0], log=True, range=MET_range[idx],histtype='step', color=ibm_color_blind_palette[0], bins=100, label='NuGun')
    axs[0, 0].hist(real_data[:,0], log=True, range=MET_range[idx], histtype='step', color=ibm_color_blind_palette[1], bins=100, label='real')
    axs[0, 0].hist(injected_dataset[:,0], log=True, range=MET_range[idx], histtype='step', color=ibm_color_blind_palette[2],bins=100, label=f'{sig} injected')
    axs[0, 0].legend()
    axs[0, 0].set_title('MET')

    # Plot for p_T e/gamma
    axs[0, 1].hist(NuGun[:,3], log=True, histtype='step', color=ibm_color_blind_palette[0],bins=100, label='NuGun')
    axs[0, 1].hist(real_data[:,3], log=True, histtype='step', color=ibm_color_blind_palette[1], bins=100, label='real data')
    axs[0, 1].hist(injected_dataset[:,3], log=True, histtype='step', color=ibm_color_blind_palette[2], bins=100, label=f'{sig} injected')
    axs[0, 1].legend()
    axs[0, 1].set_title('p_T e/gamma')

    # Plot for p_T muon
    axs[1, 0].hist(NuGun[:,39], log=True, histtype='step', color=ibm_color_blind_palette[0],bins=100, label='NuGun')
    axs[1, 0].hist(real_data[:,39], log=True, histtype='step',color=ibm_color_blind_palette[1], bins=100, label='real data')
    axs[1, 0].hist(injected_dataset[:,39], log=True, histtype='step', color=ibm_color_blind_palette[2], bins=100, label=f'{sig} injected')
    axs[1, 0].legend()
    axs[1, 0].set_title('p_T muon')

    # Plot for p_T jet
    axs[1, 1].hist(NuGun[:,63], log=True, histtype='step', color=ibm_color_blind_palette[0],bins=100, label='NuGun')
    axs[1, 1].hist(real_data[:,63], log=True, histtype='step',color=ibm_color_blind_palette[1], bins=100, label='real data')
    axs[1, 1].hist(injected_dataset[:,63], log=True, histtype='step', color=ibm_color_blind_palette[2], bins=100, label=f'{sig} injected')
    axs[1, 1].legend()
    axs[1, 1].set_title('p_T jet')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.savefig(f'leading_order_plots/all_nugun_realdata_{sig}_{MET_range[idx]}')
    plt.show()