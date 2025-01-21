import mne
import math
import pandas as pd
import seaborn as sns
import microstates.micro_constants as cs
import base.base_constants as b_cs
import numpy as np
import matplotlib.pyplot as plt
from pycrostates.io import ChData
from sklearn.preprocessing import normalize
from pycrostates.cluster import ModKMeans
from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score)
import os
import feather
import base.files_in_out as files_in_out
import platform
#import feather


class MicroManager:
    def __init__(self, files, condition=None):
        self.epo = mne.read_epochs(files.condition_files[0])
        if condition is not None:
            self.epo = self.epo[condition]
            # If condition is specified, only inlcude condition in epo
            
        self.report = mne.Report()
        self.files = files
        self.files.get_info(start_fix=26, end_fix=-14)
        self.scores = {'Silhouette': np.zeros(len(cs.cluster_numbers)),
                       'Calinski-Harabasz': np.zeros(len(cs.cluster_numbers)),
                       'Dunn': np.zeros(len(cs.cluster_numbers)),
                       'Davies-Bouldin': np.zeros(len(cs.cluster_numbers))
                       }
        self.GEV = list()
        self.figures = list()
        self.captions = list()
        self.figures_corr = list()
        self.caption_corr = list()
        self.cl_cent = []
    def preproc_epo(self):
        # select epo
        epo_filt = self.epo[cs.select_epo]
        epo_filt = epo_filt[cs.sys_mask]
        # crop epos
        self.epo_gfp = epo_filt.crop(cs.time_lim[0], cs.time_lim[1])
        self.report.add_epochs(self.epo_gfp, psd=False,
                               title='epochs filtered')

    def gfp_extraction(self):
        self.gfp_max, idx = extract_gfp_peaks(self.epo_gfp, min_peak_distance=1, last_gfp_peak=True)

    def get_epo_gfp(self):
        # transform array from gfp to epoch structure
        data_epo_out = self.gfp_max.get_data()
        size = data_epo_out.shape
        data_epo_out = np.reshape(data_epo_out.T, (size[1], size[0], 1))  # n_epochs, n_channels, n_times
        self.epo_out = mne.EpochsArray(data_epo_out, self.epo_gfp.info, events=self.epo_gfp.events,
                                       event_id=self.epo_gfp.event_id, metadata=self.epo_gfp.metadata)
        epo_out = self.epo_out
        return epo_out

    def save_gfp(self):
        type_sig = 'prestate'
        file_end = 'gfp_epo.fif'
        output_filename = self.files.out_filename(type_sig=type_sig, file_end=file_end)
        self.epo_out.save(output_filename, overwrite=True)

    def get_epo_cluster_centers(self):
        data_epo_out = self.ModK.cluster_centers_
        size_cl = data_epo_out.shape
        data_epo_out = np.reshape(data_epo_out, (size_cl[0], size_cl[1], 1))  # n_epochs, n_channels, n_times
        cl_cent_epo_out = mne.EpochsArray(data_epo_out, self.epo.info)
        return cl_cent_epo_out

    def save_single_clus_center(self, clus_cent):
        type_sig = 'prestate'
        file_end = 'final_clus-epo.fif'
        output_filename = self.files.out_filename(type_sig=type_sig, file_end=file_end)
        clus_cent.save(output_filename, overwrite=True)

    def get_subplot_axes(self, n_clusters):
        if n_clusters < 5:
            n_cols = n_clusters
            n_rows = 1
            fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
        else:
            if n_clusters % 2 == 0:

                n_cols = int(n_clusters / 2)
                n_rows = 2
                fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
            else:
                n_cols = math.ceil(int(n_clusters / 2)) + 1
                n_rows = 2
                fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
                ax = ax.flat
                fig.delaxes(ax[-1])
                ax = ax[:-1]
        return fig, ax


    def grand_cluster_analysis(self, clus_list,micro_type,fig_subj_list=None, caption_list=None):
        # set path here
        cwd=os.getcwd()
        # included this if-loop to be able to call the function multiple times in succession
        if cwd[len(cwd)-12:len(cwd)] != 'ana\\prestate':
            os.chdir(cs.prestate_path)
            
        cluster_centers = np.hstack(clus_list)
        self.report = mne.Report()
        self.cluster_centers = ChData(cluster_centers, self.epo.info)
        for k, n_clusters in enumerate(cs.cluster_numbers):
            # k is the count of the iteration of the loop, n_clusters is the value at that iteration
            ModK_gca = ModKMeans(n_clusters=n_clusters, random_state=42, n_init=1000)
            ModK_gca.fit(self.cluster_centers, n_jobs=-1)
            self.ModK = ModK_gca

            # save clus center
            cl_cent = self.get_epo_cluster_centers()
            cl_cent_fp = cs.cluster_centers_epochs_path + f'{micro_type}/cfa_n_vep_prestate_{micro_type}_cluster_centers_{n_clusters}_epo.fif'
            cl_cent.save(cl_cent_fp, overwrite=True)

            # add scores
            self.add_scores(idx=k)
            # add gev increase plot
            self.GEV.append(self.ModK._GEV_)
            # add figure to list
            fig, ax = self.get_subplot_axes(n_clusters)

            self.ModK.plot(axes=ax)
            caption = f'GEV for {n_clusters} clus solution is: {self.ModK._GEV_}'
            caption_corr = f'corr for {n_clusters} clus solution'

            self.figures.append(fig)
            self.captions.append(caption)
            fig_heat = self.get_cross_corr()
            self.figures_corr.append(fig_heat)
            self.caption_corr.append(caption_corr)

        self.fig_score_raw = self.plot_scores_raw()
        self.fig_score = self.plot_scores()
        self.fig_gev_diff = self.plot_GEV_diff() # abcde error here - I think bc it appends GEV
        if fig_subj_list is not None:
            self.report.add_figure(fig=fig_subj_list, title=f'CLuster for each subject', caption=caption_list)
        self.report.add_figure(fig=self.figures, title=f'N clusters : {n_clusters}', caption=self.captions)
        self.report.add_figure(fig=self.figures, title=f'N clusters : {n_clusters}', caption=self.captions)
        self.report.add_figure(fig=self.figures_corr, title=f'cross corr of  : {n_clusters} clusters',
                                caption=self.caption_corr)
        self.report.add_figure(fig=self.fig_score_raw, title=f'Raw evaluation score', caption='Raw scores')
        self.report.add_figure(fig=self.fig_score, title=f'Normalized evaluation score', caption='Scores')

        self.report.add_figure(fig=self.fig_gev_diff, title=f'Increment in GEV', caption='diff in GEV')
        plt.close('all')

    def save_report(self):
        type_sig = 'prestate'
        file_end = 'prestate_report.html'
        output_filename = self.files.out_filename(type_sig=type_sig, file_end=file_end)
        self.report.save(output_filename, overwrite=True)

    def save_final_report(self,filename):

        self.report.save(filename, overwrite=True)

    def add_scores(self, idx):
        self.scores["Silhouette"][idx] = silhouette_score(self.ModK)
        self.scores["Calinski-Harabasz"][idx] = calinski_harabasz_score(self.ModK)
        self.scores["Dunn"][idx] = dunn_score(self.ModK)
        self.scores["Davies-Bouldin"][idx] = davies_bouldin_score(self.ModK)

    def plot_scores(self):
        # invert davies-bouldin scores
        self.scores["Davies-Bouldin"] = 1 / (1 + self.scores["Davies-Bouldin"])

        # normalize scores using sklearn

        scores = {score: normalize(value[:, np.newaxis], axis=0).ravel()
                  for score, value in self.scores.items()}

        # set width of a bar and define colors
        barWidth = 0.2
        colors = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F"]

        # create figure
        fig = plt.figure(figsize=(10, 8))
        # create the position of the bars on the X-axis
        x = [[elt + k * barWidth for elt in np.arange(len(cs.cluster_numbers))]
             for k in range(len(scores))]
        # create plots
        for k, (score, values) in enumerate(scores.items()):
            plt.bar(
                x=x[k],
                height=values,
                width=barWidth,
                edgecolor="grey",
                color=colors[k],
                label=score,
            )
        # add labels and legend
        plt.xlabel("Number of clusters")
        plt.ylabel("Score normalize to unit norm")
        plt.xticks(
            [pos + 1.5 * barWidth for pos in range(len(cs.cluster_numbers))],
            [str(k) for k in cs.cluster_numbers],
        )
        plt.legend()
        return fig

    def plot_GEV_diff(self):
        diff_gev = np.diff(self.GEV)
        fig = plt.figure(figsize=(10, 8))
        plt.plot(cs.cluster_numbers[1:], diff_gev, 'o')
        return fig

    def get_cross_corr(self):
        clus = self.ModK.cluster_centers_
        df = pd.DataFrame(clus.T)
        # get absolute corr matrix
        df.corr().abs()
        # plot
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(df.corr().abs(), annot=True, ax=ax, cmap='PuBu')
        return fig

    def plot_scores_raw(self):
        f, ax = plt.subplots(2, 2, figsize=(20, 20))
        for k, (score, values) in enumerate(self.scores.items()):
            ax[k // 2, k % 2].bar(x=cs.cluster_numbers, height=values)
            ax[k // 2, k % 2].set_title(score)
        plt.text(
            0.03, 0.5, "Score",
            horizontalalignment='center',
            verticalalignment='center',
            rotation=90,
            fontdict=dict(size=18),
            transform=f.transFigure,
        )
        plt.text(
            0.5, 0.03, "Number of clusters",
            horizontalalignment='center',
            verticalalignment='center',
            fontdict=dict(size=18),
            transform=f.transFigure,
        )
        return f

    def return_cluster_center(self):
        clus_cent = self.epo.get_data()
        clus_cent = np.squeeze(clus_cent).T
        return clus_cent

    def grand_cluster_analysis_short(self, clus_list, n_clusters):
        # set path here
        cluster_centers = np.hstack(clus_list)
        self.cluster_centers = ChData(cluster_centers, self.epo.info)

        ModK_gca = ModKMeans(n_clusters=n_clusters, random_state=42, n_init=1000)
        ModK_gca.fit(self.cluster_centers, n_jobs=-1)
        self.ModK = ModK_gca
    
    def fitting_analyses(self, gfp_epo, conditions):
        seg_dict = dict()
        for cond in conditions:
            segmentation = self.ModK.predict(gfp_epo[cond], reject_edges=False)
            labels=segmentation.labels
            seg_dict[cond] = segmentation.compute_parameters()
        return seg_dict,labels
    

    def format_param_df(self, list_df):
        list_df.drop(columns='unlabeled', inplace=True)
        list_df['g_num'] = list_df.index
        list_df.reset_index(drop=True, inplace=True)
        list_df['id'] = list_df.index

        long_df = pd.wide_to_long(list_df, stubnames=cs.stub_names, i='id', j='map')
        long_df.reset_index(inplace=True)
        return long_df

    def save_to_csv(self, df, filename):
        df.to_csv(filename, index=False)

    def get_param_df(self, fitting_dict,cond_type='cardiac_phase'):
        list_df_g_lvl = list()
        for g_num, dic_num in fitting_dict.items():
            for cond, param in dic_num.items():
                # assign condition based on dict key
                param_df = pd.DataFrame(param, index=[g_num])
                # invert names of column so that is compatible to wide to long function
                param_df.columns = ['_'.join(x.split('_')[::-1]) for x in param_df.columns]

                if cond_type=='cardiac_phase' or cond_type=='rsp_phase':
                    awa = cond.split('/')[0]
                    phy_cond = cond.split('/')[1]
                    param_df['awareness'] = awa
                    param_df['phy_cond'] = phy_cond
                elif cond_type=='awareness':
                    param_df['awareness'] = cond
                elif cond_type=='phy_phases':
                    card = cond.split('/')[0]
                    rsp = cond.split('/')[1]
                    param_df['cardiac_phase'] = card
                    param_df['rsp_phase'] = rsp

                #append to list
                list_df_g_lvl.append(param_df)
        # concatenate all df
        list_df = pd.concat(list_df_g_lvl)
        # format df in long format
        self.long_df = self.format_param_df(list_df)
        return self.long_df
    

    def save_long_df(self,phy_cond,n_clus,micro_type):
        df_fn=f'param df/{micro_type}/{phy_cond}_{n_clus}_clusters_tsk_n_prestate_{micro_type}.feather'
        df_fp=cs.prestate_path+'/'+df_fn
        feather.write_dataframe(self.long_df,df_fp)
        feather.write_dataframe(self.long_df,df_fp)
    
    def save_long_df_csv(self, phy_cond, n_clus, micro_type):
        df_fn = f'param df/{micro_type}/{phy_cond}_{n_clus}_clusters_tsk_n_prestate_{micro_type}.csv'
        df_fp = cs.prestate_path + '/' + df_fn
        os.makedirs(os.path.dirname(df_fp), exist_ok=True)
        self.long_df.to_csv(df_fp, index=False)

