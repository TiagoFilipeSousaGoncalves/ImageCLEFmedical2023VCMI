# Imports
import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Original directory
parser.add_argument("--original_path", type=str, default="dataset", help="Directory of the original data set.")

# New (resized) directory
parser.add_argument("--new_path", type=str, default="dataset/processed/resized/", help="Directory of the resized data set.")

# New height for the resized images
parser.add_argument("--new_height", type=int, default=224, help="New height for the resized images.")

# Parse the arguments
args = parser.parse_args()


# Get the arguments
ORIGINAL_PATH = args.original_path
NEW_PATH = args.new_path
NEW_HEIGHT = args.new_height

# Create new path if needed
if not os.path.exists(NEW_PATH):
    os.makedirs(NEW_PATH)

min_h = np.inf
min_w = np.inf
max_h = 0
max_w = 0

for f in tqdm(os.listdir(ORIGINAL_PATH)):
    if(f.endswith(".jpg") or f.endswith('.png')):

        img = cv2.imread(os.path.join(ORIGINAL_PATH, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        if h < min_h:
            min_h = h
        if w < min_w:
            min_w = w

        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

        ratio = w / h
        new_w = int(np.ceil(NEW_HEIGHT * ratio))
        new_img = cv2.resize(img, (new_w, NEW_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(NEW_PATH, f), new_img)

print(min_h, min_w)
print(max_h, max_w)
