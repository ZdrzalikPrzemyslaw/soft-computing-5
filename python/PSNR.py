import imageio
import numpy as np
from math import log10, sqrt
import csv

from matplotlib import pyplot as plt

files = ['01.bmp', '02.bmp', '03.bmp', '04.bmp', '05.bmp', '06.bmp', '07.bmp', '08.bmp', ]
dirs = ['./out/1/', './out/2/', './out/3/', './out/4/', './out/8/', './out/16/', './out/32/', './out/64/', './out/128/', ]


def PSNR(original, compressed):
    v = original - compressed
    v_2 = v ** 2
    mse = np.sum(v_2)/(512**2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 50
    max_pixel = 255.0
    psnr = 10 * log10(max_pixel**2 / mse)
    return psnr

def plot_many(values):
    fig, ax = plt.subplots()

    # We need to draw the canvas, otherwise the labels won't be positioned and
    # won't have values yet.
    fig.canvas.draw()

    labels = [ '', '1','2','3','4','8','16','32','64','128']

    for i in values:
        ax.plot(i,  **{'marker': 'o'})

    ax.set_xticklabels(labels)

    plt.show()


outputs = []
psnr_values = []
for idx, j in enumerate(files):
    psnr_values.append([])
    for i in dirs:
        or_str = "./images/" + j
        comp_str = i + j
        original = np.asarray(imageio.imread(or_str)).astype(int)
        compressed = np.asarray(imageio.imread(comp_str)).astype(int)
        psnr_values[idx].append(PSNR(original, compressed))
        outputs.append("PSNR OF " + or_str + " and " + comp_str + " = " + PSNR(original, compressed).__str__())

psnr_values = np.asarray(psnr_values)

plot_many(psnr_values)

with open("PSNR.csv", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerows(psnr_values)

with open('PSNR.txt', 'w') as f:
    for line in outputs:
        f.write(f"{line}\n")