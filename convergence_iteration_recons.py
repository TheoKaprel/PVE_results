import matplotlib.pyplot as plt
import numpy as np
import itk
import glob
import click
import json

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
@click.option("--json",'jsons', multiple = True)
def plot_conv_iter_recons(source, jsons):

    source_array = itk.array_from_image(itk.imread(source))
    norm_src_array = normalize(source_array)

    if len(jsons)>0:
        path_iterations = ()
        for json_file in jsons:
            iterations_info = open(json_file).read()
            iterations_info = json.loads(iterations_info)



    fig_err, ax_err = plt.subplots()

    for (path_id,json_file) in enumerate(jsons):
        iterations_info = open(json_file).read()
        iterations_info = json.loads(iterations_info)
        path = iterations_info['iterations_path']
        n_subsets = iterations_info['n_subsets']
        beta = iterations_info['beta']


        list_img, list_iter = get_list_of_iter_img(path)
        list_iter = [it*n_subsets for it in list_iter]
        list_err = []
        legend = f'{n_subsets} SS  | beta = {beta}'

        for img_file, iter in zip(list_img,list_iter):

            img_array = itk.array_from_image(itk.imread(img_file))
            norm_img_array = normalize(img_array)
            list_err.append(np.sqrt(np.mean((norm_img_array-norm_src_array)**2) / np.mean(abs(norm_src_array**2))))

        iter_err = zip(list_iter,list_err)
        iter_err = sorted(iter_err, key=lambda x: x[0])
        iter, err = zip(*iter_err)

        ax_err.plot(iter, err, marker='o',linewidth = 1.5, label = legend)


    plt.legend()
    plt.show()







if __name__=='__main__':
    plot_conv_iter_recons()