import numpy as np
import imageio
import os


    

def img_to_black(img, threshold=50):
    """Helper function to binarize greyscale images with a cut-off."""
    img = img.astype(np.float32)
    idx = img[:, :] > threshold
    idx_0 = img[:, :] <= threshold
    img[idx] = 1
    img[idx_0] = 0
    return img

dirname = "edges/"
dirname = "midlines/"
dirname = "../training/groundtruth/"
fnames = os.listdir(dirname)
a = np.zeros((400, 400))
for f in fnames:
    img = imageio.imread(dirname+f)
    img = img_to_black(img)
    a += img
print(a.sum() / 100 / 400 / 400)
