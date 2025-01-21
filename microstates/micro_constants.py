import os
import sys
import pandas as pd

#sys.path.append('C:/Users/User/Dropbox/PC (2)/Documents/GitHub/BBC')


# define starting datafolder

#this is the directory
import os
import platform



platform.system()

# define starting datafolder

if platform.system()=='Darwin':
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk')
else:
    os.chdir('Z:/BBC/WP1/data/EEG/tsk')
    #os.chdir('c:/Users/Engi/all/BBC/WP1/data/EEG/tsk')


#os.chdir('C:/Users/User/Dropbox/PC (2)/Documents/tsk_master')
#base_datafolder = '/Volumes/Elements/'
#this is where the data is

end_format='cfa_vep_clean_epo.fif'
eeg_exp='tsk'
datafolder='preproc'
cond='n'
end_format_cluster='cfa_vep_final_clus-epo.fif'
end_format_gfp_epo='cfa_vep_gfp_epo.fif' # this is the epo with only the gfp max

## preproc constants
select_epo='correct/normal'
sys_mask='sys_mask==1'
time_lim=[-0.05,0]

##cluster analysis
cluster_numbers=range(2,10)
prestate_path='ana/prestate'
prestate_clus_selection_final_fn = 'prestate_clus_selection_final.csv'
prestate_clus_selection_final_fp=prestate_path+'/'+prestate_clus_selection_final_fn
prestate_clus_selection_condgfp_final_fn = 'prestate_clus_selection_condgfp_final.csv'
prestate_clus_selection_condgfp_final_fp = prestate_path+'/'+prestate_clus_selection_condgfp_final_fn

# name for clusters conducted in 2 steps, first cluster at subject level and then at group level
report_fn_final = 'cfa_n_vep_prestate_subclust.html'

# name for clusters conducted on entire data
report_all_gfp_fn_final='cfa_n_vep_prestate_whole_gfp.html'
cluster_centers_epochs_path = 'clus_center/'

# name for clusters conducted within condition
report_all_cond_gfp_final = 'cfa_n_vep_prestate_condgfp_'
# constants for combination analysis


# Define the base combinations
base_combinations = {
    'aware': ['sys', 'dia'],
    'unaware': ['sys', 'dia'],
    'sys': ['aware', 'unaware'],
    'dia': ['aware', 'unaware'],
    'inh': ['aware', 'unaware'],
    'exh': ['aware', 'unaware']
}

# Additional combinations
additional_combinations = {
    'aware': ['inh', 'exh'],
    'unaware': ['inh', 'exh'],
    'sys': ['aware', 'unaware'],
    'dia': ['aware', 'unaware'],
    'inh': ['aware', 'unaware'],
    'exh': ['aware', 'unaware']
}


fitting_combinations = {
    'aware': ['sys', 'dia'],
    'unaware': ['sys', 'dia'],
    'sys': ['aware', 'unaware'],
    'dia': ['aware', 'unaware'],
    'inh': ['aware', 'unaware'],
    'exh': ['aware', 'unaware']}
    # 'aware': ['inh', 'exh']
    # 'unaware': ['inh', 'exh']]}
# fitting analysis
n_clus_subclust=[4,6]
n_clus_wholegfp=[5]
n_clus_condgfp = [5,5,5,5,4,4]

n_clusters_list = {
    'aware': 5,
    'unaware': 5,
    'dia' : 5,
    'sys' : 5,
    'inh' : 4,
    'exh' : 4}

phy_cond=['cardiac_phase','rsp_phase','awareness']
#phy_cond=['phy_phases']

condition_list_awa=['aware','unaware']
condition_list_card=['aware/sys','unaware/sys','aware/dia','unaware/dia']
condition_list_rsp=['aware/inh','unaware/inh','aware/exh','unaware/exh']
condition_list_inh=['aware/inh','unaware/inh']
condition_list_exh=['aware/exh','unaware/exh']
condition_list_sys = ['aware/sys', 'unaware/sys']
condition_list_dia = ['aware/dia', 'unaware/dia']
condition_list_phy=['sys/inh','sys/exh','dia/inh','dia/exh']
condition_lists = {
    'aware': condition_list_awa,
    'unaware': condition_list_awa,
    'sys': condition_list_sys,
    'dia': condition_list_dia,
    'inh': condition_list_inh,
    'exh': condition_list_exh
}


all_conditions_list=['aware','unaware','sys','dia','inh','exh']

grand_conditions_list = ['aware/sys', 'aware/dia', 'aware/inh', 'aware/exh', 'unaware/sys', 'unaware/dia', 'unaware/inh', 'unaware/exh', 'dia', 'inh', 'exh']

stub_names=['corr_mean_','gev_','occurrences_','meandurs_','timecov_']
