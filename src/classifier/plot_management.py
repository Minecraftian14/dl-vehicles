import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_this(ax: plt.Axes, title=None):
    if title: ax.set_title(title)


def plot_rectangle(ax: plt.Axes, rect: tuple[float, float, float, float]):
    rect = patches.Rectangle(
        (rect[0], rect[1]),
        rect[2] - rect[0] - 1,  # -1 for inset visual
        rect[3] - rect[1] - 1,
        linewidth=2,
        edgecolor='blue',
        facecolor='none',
    )
    ax.add_patch(rect)


def plot_sample(descriptor, image, ax: plt.Axes = None, addon=''):
    if ax is None: _, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(image.astype(np.uint8))
    plot_rectangle(ax, descriptor['box'])
    plot_this(ax, descriptor['label'] + f' ({image.shape[1]}x{image.shape[0]}) {addon}')