import matplotlib.pyplot as plt
import numpy as np
import itk
import glob
import click

def get_list_of_iter_img(path_iterations):
    id_d = path_iterations.find("%d")
    list_imgs = glob.glob(f'{path_iterations[:id_d]}*{path_iterations[id_d+2:]}')
    list_iter = [int(img[id_d:img.find(path_iterations[id_d+2:])]) for img in list_imgs]
    return list_imgs,list_iter

def normalize(array):
    return array / array.sum()

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-s', '--source',)
@click.option('-p', 'path_iterations', help="path to reconstructed images with %d instead of iteration number. ex rec_img_120_%d.mha")
@click.option('--slice')
@click.option('--profile')
def plot_conv_iter_recons(source, path_iterations, slice, profile):

    source_array = itk.array_from_image(itk.imread(source))
    norm_src_array = normalize(source_array)

    list_img, list_iter = get_list_of_iter_img(path_iterations)
    list_err = []

    if (slice is not None) and (profile is not None):
        fig_prof,ax_prof = plt.subplots()
        do_plot_prof = True
        slice,profile = int(slice), int(profile)
        ax_prof.plot(norm_src_array[slice,profile,:], label = 'src')
    else:
        do_plot_prof = False

    for img_file, iter in zip(list_img,list_iter):

        img_array = itk.array_from_image(itk.imread(img_file))
        norm_img_array = normalize(img_array)
        list_err.append(np.mean((norm_img_array-norm_src_array)**2) / np.mean(abs(norm_src_array)))
        # list_err.append(np.mean(np.abs(norm_img_array-norm_src_array)) / np.mean(abs(norm_src_array)))

        if do_plot_prof:
            ax_prof.plot(norm_img_array[slice,profile,:], label = f"iter {iter}")


    plt.legend()


    fig_err,ax_err = plt.subplots()
    ax_err.scatter(list_iter, list_err, s=10, marker='o')





    plt.show()







if __name__=='__main__':
    plot_conv_iter_recons()