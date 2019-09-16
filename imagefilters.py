# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:03:21 2019

@author: Yonatan
"""

import numpy as np
import argparse
import cv2
from skimage.exposure import rescale_intensity
import os, pathlib


class Filter: 
    def __init__(self, filters):
        
        self.filters = filters

        self.KernelBank =  {"small_blur":np.ones((7, 7), dtype="float") * (1.0 / (7 * 7)), 
                            "large_blur":np.ones((21, 21), dtype="float") * (1.0 / (21 * 21)),
                            "sharpen": np.array(([0, -1, 0], 
                                                 [-1, 5, -1], 
                                                 [0, -1, 0]), dtype="int"), 
                            "laplacian":np.array(([0, 1, 0],
                                                  [1, -4, 1], 
                                                  [0, 1, 0]), dtype="int"),
                            "sobel_x":np.array(([-1, 0, 1], 
                                                [-2, 0, 2], 
                                                [-1, 0, 1]), dtype="int"),
                            "sobel_y":np.array(([-1, -2, -1],
                                                [0, 0, 0], 
                                                [1, 2, 1]), dtype="int"),
                            "emboss":np.array(([-2, -1, 0], 
                                               [-1, 1, 1], 
                                               [0, 1, 2]), dtype="int")}
        
        
    def apply_filter(self, image_path):
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        
        out = self.image
        for i in self.filters:
            print("applying {}".format(i))
            out = self.convolve(out, self.KernelBank[i])
        
        print('done')
        cv2.imshow("{} - convole".format("+".join(self.filters)), out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return out

    def convolve(self, img, K):  #amount in the range (0, 1)
        (image_H, image_W) = img.shape[:2]
        (K_H, K_W) = K.shape[:2]
        padding = (K_W - 1) // 2
        image = cv2.copyMakeBorder(img, padding, padding, 
                                   padding, padding, cv2.BORDER_REPLICATE)
        
        out = np.zeros((image_H, image_W), dtype='float')
        
        
        for y in np.arange(padding, image_H + padding):
            for x in np.arange(padding, image_W + padding):
                center_matrix = image[y - padding:y + padding + 1,
                                      x - padding:x + padding + 1]
                center_pixel = (center_matrix * K).sum()
                out[y - padding, x - padding] = center_pixel
                
        out = rescale_intensity(out, in_range=(0, 255))
        out = (out * 255).astype("uint8")
        
        return out
    

def main():
    ap = argparse.ArgumentParser() 
    ap.add_argument("-i", "--image", required=True, type=str, help="path to the input image") 
    ap.add_argument("-f", "--filter", nargs='+', default=[], help="supported: small_blur, large_blue, sharpen, laplacian, sobel_x, sobel_y, emboss")
    args = ap.parse_args()
    f = Filter(args.filter)

    f.apply_filter(args.image)
    
main()




            
            
            
            
            
            
    
    
