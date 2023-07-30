import matplotlib.pyplot as plt
import numpy as np

# def plot_2d_vis(results, title, cluster_colors):
#     """ Plots the first two components of the results dictionary as a grid of scatter plots."""
#     num_rows = len(results.keys())
#     num_cols = len(list(results.values())[0].keys())
#     fig, axs = plt.subplots(
#         num_rows,
#         num_cols,
#         figsize=(num_rows * 5, num_cols * 5),
#     )
#     if axs.ndim == 1:
#         axs = axs.reshape(1, -1)
#     fig.suptitle(title, fontsize=25)
#     for i, (norm_key, sub_dict) in enumerate(results.items()):
#         for j, (trans_key, result) in enumerate(sub_dict.items()):
#             axs[i, j].scatter(
#                 result[:, 0],
#                 result[:, 1],
#                 s=7,
#                 c=cluster_colors,
#             )
#             axs[i, j].set_title(f"{norm_key} {trans_key}".title())
#             axs[i, j].set_xticks([])
#             axs[i, j].set_yticks([])


def plot_2d_vis(results, title, clusters=None):
    # plot t-SNE
    num_rows = len(results.keys())
    num_cols = len(list(results.values())[0].keys())
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 5, num_rows * 5),
    )
    if axs.ndim == 1:
        axs = axs.reshape(-1, 1)

    fig.suptitle(title, fontsize=25)
    for i, (norm_key, sub_dict) in enumerate(results.items()):
        for j, (trans_key, result) in enumerate(sub_dict.items()):
            axs[i, j].scatter(
                result[:, 0],
                result[:, 1],
                s=7,
                c=clusters[norm_key][trans_key]
                if clusters is dict
                else clusters[norm_key][trans_key],
                cmap="tab20" if clusters is dict else None,
            )
            axs[i, j].set_title(f"{norm_key} {trans_key}".title())
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

