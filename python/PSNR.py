import cv2
import numpy as np
from math import log10, sqrt

files = ['01.bmp', '02.bmp', '03.bmp', '04.bmp', '05.bmp', '06.bmp', '07.bmp', '08.bmp', ]
dirs = ['./out/1/', './out/2/', './out/3/', './out/4/', './out/8/', './out/16/', './out/32/', './out/64/', './out/128/',
        './out/256/', ]


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


outputs = []
for i in dirs:
    for j in files:
        or_str = "./images/" + j
        comp_str = i + j
        original = cv2.imread(or_str)
        compressed = cv2.imread(comp_str)
        PSNR(original, compressed)
        outputs.append("PSNR OF " + or_str + " and " + comp_str + " = " + PSNR(original, compressed))
