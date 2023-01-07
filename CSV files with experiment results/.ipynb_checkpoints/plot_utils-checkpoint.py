import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as mpatches


def get_color_for_par(
    total_par,
    perc_25=8940354,
    perc_50=11304258,
    perc_75=11596802,
    perc_100=13753922,
):
    if total_par <= perc_25:
        return "tab:blue"
    elif perc_25 < total_par <= perc_50:
        return "lightskyblue"
    elif perc_50 < total_par <= perc_75:
        return "orange"
    else:
        return "red"

def get_color_config(
    feature_val,
    perc_25=32,
    perc_50=64,
    perc_75=128,
    perc_100=256,
):
    if feature_val <= perc_25:
        return "tab:blue"
    elif perc_25 < feature_val <= perc_50:
        return "lightskyblue"
    elif perc_50 < feature_val <= perc_75:
        return "orange"
    else:
        return "lightgray"


def plot_ppl_flops_model_size(
    df_eval,
):
    f, ax = plt.subplots(figsize=(12, 8), dpi=128)
    ax.set(xscale="log")#, yscale="log")

    #
    for run_name in df_eval.run_name.unique():
        df_i = df_eval.loc[df_eval.loc[:, 'run_name'] == run_name, :]
        #sns.set_style("darkgrid")
        sns.lineplot(
            data=df_i,
            x="FLOPS Hoffman total",
            y="eval/perplexity",
            lw=4,
            alpha=0.7,
            ax=ax, 
            color=get_color_for_par(df_i.loc[:, 'Total parameters'].unique()),
            #scatter_kws={"s": 10},
        )


    #
    handles = []
    prev_size_ = 0
    for val_idx, val in enumerate([8940354, 11304258, 11596802, 13753922]):

        # col
        col_ = get_color_for_par(val)
        #patch.set_color(col_)

        # label
        size_ = round(val/10**6)
        label_ = f"{prev_size_} < Num. par. <= {size_} Mil."
        prev_size_ = size_

        #
        handles.append(mpatches.Patch(color=col_, label=label_))

    #
    ax.legend(handles=handles)
    plt.xlabel("FLOPs (based on Hoffman et.al.)")
    plt.ylabel("Perplexity on development set")
    plt.rcParams.update({'font.size': 22})
    plt.savefig("./PLOTS/PPL_vs_FLOPs.jpg")
    
    return


def plot_ppl_flops_model_config_feature(
    df_eval,
    config_feature="embedding_size",
    config_feature_vals=[32, 64, 128, 256],
):
    f, ax = plt.subplots(figsize=(12, 8), dpi=128)
    ax.set(xscale="log")#, yscale="log")


    for run_name in df_eval.run_name.unique():
        df_i = df_eval.loc[df_eval.loc[:, 'run_name'] == run_name, :]
        #sns.set_style("darkgrid")
        sns.lineplot(
            data=df_i,
            x="FLOPS Hoffman total",
            y="eval/perplexity",
            lw=4,
            alpha=0.7,
            ax=ax, 
            color=get_color_config(
                feature_val=df_i.loc[:, config_feature].unique(),
                perc_25=config_feature_vals[0],
                perc_50=config_feature_vals[1],
                perc_75=config_feature_vals[2],
                perc_100=config_feature_vals[3],
            #scatter_kws={"s": 10},
            )
        )
    
    
    #
    feat2legend = {
        "embedding_size": "Emb. size",
        "hidden_size": "Hidden size",
        "intermediate_size": "Inter. size",
        "num_attention_heads": "Attn. heads",
        "num_hidden_layers": "Hidden layers",
    }

    #
    handles = []
    prev_size_ = 0
    for val_idx, val in enumerate(config_feature_vals):

        # col
        col_ = get_color_config(
            feature_val=val,
            perc_25=config_feature_vals[0],
            perc_50=config_feature_vals[1],
            perc_75=config_feature_vals[2],
            perc_100=config_feature_vals[3],
        )
        #patch.set_color(col_)

        # label
        size_ = round(val)
        legend_name = feat2legend[config_feature]
        label_ = f"{legend_name} = {size_}"
        prev_size_ = size_

        #
        handles.append(mpatches.Patch(color=col_, label=label_))

    #
    ax.legend(handles=handles)
    plt.xlabel("FLOPs (based on Hoffman et.al.)")
    plt.ylabel("Perplexity on development set")
    plt.rcParams.update({'font.size': 22})
    plt.savefig(f"./PLOTS/PPL_vs_FLOPs_by_{config_feature}.jpg")
    
    return