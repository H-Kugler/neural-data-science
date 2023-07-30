import matplotlib.pyplot as plt
import numpy as np

def plot_2d_vis(results, title, clusters=None, transpose=False):
    # plot t-SNE
    n_norms = len(results.keys())
    n_trans = len(list(results.values())[0].keys())
    num_rows = n_norms if not transpose else n_trans
    num_cols = n_trans if not transpose else n_norms
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 5, num_rows * 5),
    )
    if axs.ndim == 1 and transpose:
        axs = axs.reshape(1, -1)
    elif axs.ndim == 1 and not transpose:
        axs = axs.reshape(-1, 1)

    fig.suptitle(title, fontsize=25)
    for i, (norm_key, sub_dict) in enumerate(results.items()):
        for j, (trans_key, result) in enumerate(sub_dict.items()):
            row, col = (i, j) if not transpose else (j, i)
            axs[row, col].scatter(
                result[:, 0],
                result[:, 1],
                s=7,
                c=clusters[norm_key][trans_key]
                if type(clusters) is dict
                else clusters,
                cmap="tab20" if type(clusters) is dict else None,
            )
            axs[row, col].set_title(f"{norm_key} {trans_key}".title())
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])