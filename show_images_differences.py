#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import itk

def main():
    print(args)
    # Generate synthetic images for demonstration (replace with actual images)
    reference_image = itk.array_from_image(itk.imread(args.source))[args.slice,:,:]
    images = [itk.array_from_image(itk.imread(fn))[args.slice,:,:] for fn in args.images]
    # Compute differences
    differences = [img - reference_image for img in images]

    # Plot the images
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    # Line 1: Reference and images
    vmin = min(img.min() for img in [reference_image] + images)
    vmax = max(img.max() for img in [reference_image] + images)

    # Plot reference image
    axes[0, 0].imshow(reference_image, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
    axes[0, 0].set_title('Reference')
    axes[0, 0].axis('off')

    # Plot the 5 images
    for i, img in enumerate(images):
        ax = axes[0, i + 1]
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(f'Image {i + 1}')
        ax.axis('off')

    # Remove the last unused subplot in the first row
    axes[0, -1].axis('off')

    # Line 2: Differences
    vmin_diff = min(diff.min() for diff in differences)
    vmax_diff = max(diff.max() for diff in differences)
    norm = TwoSlopeNorm(vmin=vmin_diff, vcenter=0, vmax=vmax_diff)

    # Make the lower-left subplot empty
    axes[1, 0].axis('off')

    # Plot differences
    for i, diff in enumerate(differences):
        ax = axes[1, i + 1]
        im = ax.imshow(diff, cmap='coolwarm', norm=norm, interpolation='none')
        ax.set_title(f'Diff {i + 1}')
        ax.axis('off')

    # Add colorbar to the last subplot in the second row
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, orientation='vertical')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--source")
    parser.add_argument("--slice", type = int)
    args = parser.parse_args()
    main()
