# src/graph_utils.py

import os
import matplotlib.pyplot as plt

def save_and_show(fig, filename, show=True):
    """
    Save matplotlib figure to graphs/ directory and optionally show it.

    Args:
        fig: matplotlib figure object
        filename: name of the file (e.g., 'sentence_length.png')
        show: whether to display the figure inline
    """
    graphs_dir = os.path.join(os.path.dirname(__file__), "..", "graphs")
    graphs_dir = os.path.abspath(graphs_dir)
    os.makedirs(graphs_dir, exist_ok=True)

    path = os.path.join(graphs_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved graph → {path}")
