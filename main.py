import funcs
import numpy as np
from PIL import Image

img1 = np.array(Image.open("box.png"))
img2 = np.array(Image.open("box_in_scene.png"))
funcs.apply(img1, img2)
