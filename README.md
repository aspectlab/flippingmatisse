#flippingmatisse

This code repository contains experimental code used to determine possible "flips" in the Matisse lithograph dataset.  It does *not* contain the required image data referenced in step 2.

Steps to get going:

1. First, clone this git repository, which should give you the following:<br>
    checkpoints/         -- empty directory, stores checkpoints during training<br>
    data/                -- empty directory, place to put the image data<br>
    create_datafiles.py  -- Python script to generate pre-processed data files from raw images<br>
    flipping.ipynb       -- Main script<br>
    README.md            -- this file<br>
    saved_model.hdf5     -- provided pre-trained model<br>

2. If you were provided the 860 *greyscale* Matisse images, put them in a folder called ./data/matisse_grey/, and you're done with this step.  Otherwise, if you were provided the raw color Matisse TIF images, put those 860 files in the folder ./data/matisse/.  Then, crop and convert them to greyscale with the following commands (requires ImageMagick) --
mkdir data/matisse_grey/
cd data/matisse/
mogrify -format png -resize 1836x1536 -colorspace gray -crop 1024x1024+406+256 -define png:bit-depth=8 *.tif
mv *.png ../matisse_grey/

3. Run create_datafiles.py to generate matisse.npz and matisse_aug_valid.npz.  You may delete the directories "matisse_tiled" and "matisse_aug_valid" which are created as part of this process.

4. Run flipping.ipynb which will produce results based on a pre-trained model.  To train your own model instead, set "TRAIN_MODE = True".  Start fiddling!

Authors: L. Lackey, A. Grootveld, K. Aguilar, A.G. Klein
