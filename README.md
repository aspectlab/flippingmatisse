# flippingmatisse

This code repository contains experimental code used to determine possible "flips" in the Matisse lithograph dataset.  It does *not* contain the required image data referenced in step 2.  

## Initial setup
1. First, clone this git repository, which should give you the following:<br>
    `checkpoints/`         -- empty directory, stores checkpoints during training<br>
    `data/`                -- empty directory, place to put the image data<br>
    `create_datafiles.py`  -- Python script to generate pre-processed data files from raw images<br>
    `flipping.ipynb`       -- Main script for generating feature vectors (Jupyter version)<br>
    `flipping.py`          -- Main script for generating feature vectors (Python version)<br>
    `README.md`            -- this file<br>
    `saved_model.hdf5`     -- provided pre-trained model<br>

2. If you were provided the 860 *greyscale* Matisse images, put them in a folder called `./data/matisse_grey/`, and you're done with this step.  Otherwise, if you were provided the raw *color* Matisse TIF images, put those 860 files in a folder called `./data/matisse/`.  Then, crop and convert them to greyscale with the following commands (requires ImageMagick) --<br>
`mkdir data/matisse_grey/`<br>
`cd data/matisse/`<br>
`mogrify -format png -resize 1836x1536 -colorspace gray -crop 1024x1024+406+256 -define png:bit-depth=8 *.tif`<br>
`mv *.png ../matisse_grey/`

3. Run create_datafiles.py to generate `matisse.npz` and `matisse_aug_valid.npz`.  You may optionally delete the directories `data/matisse_tiled` and `data/matisse_aug_valid` which are created as part of this process.

## Compute feature vectors and distance matrix (Python)
This step is done using Python code which makes use of TensorFlow.  You can simply run `flipping.ipynb` (or flipping.py) which will generate feature vectors and a distance matrix based on a pre-trained model.  The results are stored in a Matlab file called `output.mat`.  To train your own model instead, adjust the flag `TRAIN_MODE = True` at the top of the Python code. 

## Use distance matrix to compute suggested flips, and generate original/flipped mosaics (Matlab)
This step is done using Matlab.  Assuming you have followed the prior step and generated the distance matrix, a list of suggested flips can be generated via the commands:<br>
`load output.mat`<br>
`[uniqueEds, editions, flip] = find_flips(D, fnames);`<br>
`display_flips(flip,fnames)`<br>

Then, you can batch generate image mosiacs comparing original/flipped versions by typing:<br>
`mkdir mosaics`<br>
`tile_shuffle('data/matisse_grey/', 'mosiacs/', fnames, uniqueEds, editions, flip)`

Authors: L. Lackey, A. Grootveld, K. Aguilar, A.G. Klein
