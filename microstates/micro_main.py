#%%
import microstates.micro_helper as hp
import microstates.micro_constants as cs
import base.files_in_out as files_in_out
import base.base_constants as b_cs  # all g_num
import mne

#%% Do not run right now!!!

fig_list=list()
clus_list=list()
caption_list=list()
for g_n in b_cs.G_N_prestate:
    files = files_in_out.GetFiles(cs.datafolder, g_num=g_n,
                                  eeg_format=cs.end_format, condition=cs.cond)
    # saving files in "files"

    MicroObject = hp.MicroManager(files)
    MicroObject.preproc_epo()
    MicroObject.gfp_extraction()
    MicroObject.get_epo_gfp()




#%% analyses on whole data
list_gfp_epo=list()
for g_n in b_cs.G_N_prestate:
    files = files_in_out.GetFiles(cs.datafolder, g_num=g_n,
                                  eeg_format=cs.end_format_gfp_epo, condition=cs.cond)

    MicroObject = hp.MicroManager(files)
    list_gfp_epo.append(MicroObject.return_cluster_center())

MicroObject.grand_cluster_analysis(list_gfp_epo,micro_type='wholegfp')
MicroObject.save_final_report(filename=cs.report_all_gfp_fn_final)

#%% Fitting on whole data
list_gfp_epo=list()
for g_n in b_cs.G_N_prestate:
    files = files_in_out.GetFiles(cs.datafolder, g_num=g_n,
                                  eeg_format=cs.end_format_gfp_epo, condition=cs.cond)

    MicroObject = hp.MicroManager(files)
    list_gfp_epo.append(MicroObject.return_cluster_center())
    
for n_clus in cs.n_clus_wholegfp:
    MicroObject.grand_cluster_analysis_short(list_gfp_epo,n_clusters=n_clus)
    #define which condition to run
    for phy_cond in cs.phy_cond:
        if phy_cond=='cardiac_phase':
            cond_list=cs.condition_list_card
        elif phy_cond=='rsp_phase':
            cond_list=cs.condition_list_rsp
        elif phy_cond=='awareness':
            cond_list=cs.condition_list_awa
        elif phy_cond=='phy_phases':
            cond_list=cs.condition_list_phy

        fitting_dict=dict()
        for g_n in b_cs.G_N_prestate:
            files = files_in_out.GetFiles(cs.datafolder, g_num=g_n,
                                          eeg_format=cs.end_format_gfp_epo, condition=cs.cond)
            gfp_epo=mne.read_epochs(files.condition_files[0])
            fitting_dict[g_n],label=MicroObject.fitting_analyses(gfp_epo,conditions=cond_list)
        # returns formatted df form dict into long format
        long_df=MicroObject.get_param_df(fitting_dict,cond_type=phy_cond)
        MicroObject.save_long_df(phy_cond,n_clus,micro_type='wholegfp')


#%% Second try - clustering per condition (to find optimal solution)
for phys_cond in cs.all_conditions_list:
    list_gfp_epo_cond=list()
    for g_n in b_cs.G_N_prestate:
        files = files_in_out.GetFiles(cs.datafolder, g_num=g_n,
                                      eeg_format=cs.end_format_gfp_epo, condition=cs.cond)
    
        MicroObject = hp.MicroManager(files, condition=phys_cond)
        list_gfp_epo_cond.append(MicroObject.return_cluster_center())
    
    MicroObject.grand_cluster_analysis(list_gfp_epo_cond,micro_type='condgfp_'+phys_cond)
    MicroObject.save_final_report(filename=cs.report_all_cond_gfp_final+phys_cond+'.html')


