#!/usr/bin/env python3
from PIL import Image
import numpy as np

im = Image.open("output/systematic_evaluation_30casesnew/case_01_row1325_T2W_comparison.png")
print("Image size (W, H):", im.size)
