
import cv2
import os
import glob

def resize_img(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(size,size), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(image_path,img)


EXTENSION = 'JPEG'


root = os.path.expanduser("./data/tiny-imagenet-200")
split = "val"
split_dir = os.path.join(root, split)
image_paths = sorted(glob.iglob(os.path.join(split_dir, '**', '*.%s' % EXTENSION), recursive=True))

for path in image_paths:
    resize_img(path, 224)

# https://github.com/tjmoon0104/Tiny-ImageNet-Classifier


