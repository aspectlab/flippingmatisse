#!/usr/bin/env python
# coding: utf-8

# # Triplet Network for Identifying "Flips" in Matisse dataset
# 

# **version 0.1** -- 05/04/2021 -- AGK -- initial version<br>
# **version 0.2** -- 05/11/2021 -- AGK -- scaled features to 0.5 radius hypersphere, permuted output file to adopt Yale ICPH ordering

# ## Import necessary libraries

# In[ ]:


import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics.pairwise import euclidean_distances
from tabulate import tabulate
from IPython.display import clear_output


# ## Tweakable parameters

# In[ ]:


TRAIN_MODE = False                 # set to True if we need to train, set to False to load pre-trained model
TRAINDATA_FILE = 'data/matisse_aug_valid.npz'  # file containing training and validation data set
TESTDATA_FILE =  'data/matisse.npz'   # file containing test data set
MODEL_FILENAME = 'fm_model'        # file to load pre-trained model (if TRAIN_MODE=False)
EMB_SIZE = 16                      # num elements in feature vector / embedding output of NN
BATCH_SIZE = 512                   # size of each batch
EPOCHS = 400                       # number of epochs to run
ALPHA = 0.5                        # triplet loss parameter
L2NORM = True                      # indicates whether embeddings are L2 normalized to hypersphere
JUPYTER = False                    # true for live loss plotting (.ipynb), false to save plots to files (.py)


# ## Define functions to assist with triplet data generation and triplet loss

# In[ ]:


# function to create a batch of triplets
def create_batch(x, y, tiles_per_image, batch_size=256):
    x_anchors = np.zeros((batch_size, x.shape[1], x.shape[1],1))
    x_positives = np.zeros((batch_size, x.shape[1], x.shape[1],1))
    x_negatives = np.zeros((batch_size, x.shape[1], x.shape[1],1))
    
    for i in range(0, batch_size):
        # Pick a random anchor
        random_index = random.randint(0, x.shape[0] - 1)  # pick a random index for anchor
        x_anchors[i] = x[random_index]                    # grab anchor image
        
        # grab indices of all images in same class, and those not in same class
        indices_for_pos = np.squeeze(np.where(y == y[random_index]))
        startIdx=np.int64(np.floor(random_index/tiles_per_image/4)*tiles_per_image*4)
        indices_for_neg = np.setdiff1d(np.arange(startIdx, startIdx+tiles_per_image*4), indices_for_pos, assume_unique=True)
        
        # pick random indices for other tile of same class (positive) and one not in same class (negative)
        x_positives[i] = x[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negatives[i] = x[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
    
    return [x_anchors, x_positives, x_negatives]


# data generator functions
def data_generator(x, y, tiles_per_image, batch_size=256):
    while True:
        xx = create_batch(x, y, tiles_per_image, batch_size)
        yy = np.zeros((batch_size, 3*EMB_SIZE))  # since loss function doesn't depend on classes, set to zero
        yield xx, yy


# triplet loss function
def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:EMB_SIZE], y_pred[:,EMB_SIZE:2*EMB_SIZE], y_pred[:,2*EMB_SIZE:]
    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + ALPHA, 0.)


# ## Define functions to assist with metrics computation

# In[ ]:


def compute_image_distances(embs, tiles_per_image):
    # embs contains the features for each tile, tile_per_image contains the # of tiles per image in the embs
    
    # define some intermediate variables
    tile_distances = euclidean_distances(embs)
    num_classes = len(embs)//tiles_per_image
    
    # calculate per-image distances (approach #1) using sum of total per-tile distances
    img_distances1 = tile_distances.reshape(num_classes,tiles_per_image,num_classes,tiles_per_image).sum(axis=(1,3))  
    img_distances1 = img_distances1 + np.diag(np.diag(img_distances1))/(tiles_per_image-1) # correct for fact that block diagonals add up TILES_PER_IMAGE fewer distances
    
    # calculate per-image distances (approach #2) using euclidean difference of centroids
    centroids = np.zeros((num_classes, EMB_SIZE))
    for i in range(num_classes):
        centroids[i]=embs[range(i*tiles_per_image, (i+1)*tiles_per_image),:].mean(axis=0)/2
    img_distances2 = euclidean_distances(centroids)
    
    return [img_distances1, img_distances2, centroids]

def mean_self_similarity_rank(D):
    vals = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        vals[i] = np.where(np.argsort(D[:,i]) == i)[0].item()
    return np.mean(vals)/(D.shape[0]-1)  # number between 0 (good) and 1 (bad)

def retrieval_metrics(D, grps, epg=10, digits=2):
    pat1=0
    mrr=0
    map=0
    for k in range(grps):  # loop over similarity groups
        for i in range(epg*k,epg*(k+1)):  # loop over each element in similarity group
            dist = np.delete(np.stack([D[i,:], np.arange(0,D.shape[0])],axis=1),i,axis=0)
            T = dist[np.argsort(dist[:,0]),1]
            g = np.nonzero(np.logical_and(T>=epg*k, T<epg*(k+1)))[0]    # get rank of similar images (i.e. those with same label)
            pat1 = pat1+(g[0]==0)/grps/epg;
            mrr = mrr+1/(1+g[0])/grps/epg;
            map = map+np.mean(np.arange(1,epg)/(g+1))/grps/epg;
    
    return [round(pat1*100,digits), round(mrr*100,digits), round(map*100,digits)]


# ## Load training and validation data

# In[ ]:


npzfile = np.load(TRAINDATA_FILE)
x_train = npzfile['x_train']
y_train = npzfile['y_train']
fnames_train = npzfile['fnames_train']  # filenames corresponding to every tile in x_train (not used below, but helpful for debug)
classnames_train = npzfile['classnames_train']  # ordered image names (which subsequently appear in img_distanceX matrices, helpful for debug)
x_valid = npzfile['x_valid']
y_valid = npzfile['y_valid']
tilesize = x_train.shape[1]  # input images presumed to be square, have this number of pixels per side 
tiles_per_image = x_train.shape[0]//classnames_train.shape[0]  # number of tiles per image (assumed adjacent in data set)
valid_tiles_per_image = x_valid.shape[0]//classnames_train.shape[0]  # number of tiles per image in validation set
x_train=x_train[:,:,:,np.newaxis] # add dimension
x_valid=x_valid[:,:,:,np.newaxis]


# ## Build network for multi-GPU rig

# ### Callbacks

# In[ ]:


# Callback to compute each epoch on validation data
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, valid_tiles_per_image, embmod):
        super(Metrics, self).__init__()
        self.x, _ = valid_data
        self.valid_tiles_per_image = valid_tiles_per_image
        self.embmod=embmod
    
    def on_epoch_end(self, epoch, logs):
        # compute mean self-similarity score on validation set
        embs_val = self.embmod.predict(self.x)
        [img_distances1, _, _] = compute_image_distances(embs_val, self.valid_tiles_per_image)
        mssr = mean_self_similarity_rank(img_distances1)
        logs['val_mssr'] = mssr
        
        return

# Callback to live plot losses (in Jupyter) or save to file (in vanilla Python)
class Plotter(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.loss = []
        self.val_loss = []
        self.val_mssr = []
    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_mssr.append(logs.get('val_mssr'))
        clear_output(wait=True)
        plt.figure(figsize=(16, 5))
        plt.semilogy(np.arange(1, epoch+2), self.val_mssr,'o-')
        plt.semilogy(np.arange(1, epoch+2), self.val_loss,'o-')
        plt.semilogy(np.arange(1, epoch+2), self.loss,'o-')
        plt.grid(which='both')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss at each epoch')
        plt.legend(['validation mssr', 'validation loss', 'training loss'], loc='upper left')
        if JUPYTER:
            plt.show()
        else:
            plt.savefig('figs/{:04d}.png'.format(epoch+1))
            plt.close()


# ### Model description

# In[ ]:


# create multi-GPU setup
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # declare neural network architecture (4-layer conv net, with fully connected layer at output)
    input_img = tf.keras.layers.Input(shape=(tilesize, tilesize,1))
    next = tf.keras.layers.Conv2D(16,(5, 5), activation='relu', padding='same')(input_img)
    next = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(next)
    next = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(next)
    next = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(next)
    next = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(next)
    next = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(next)
    next = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(next)
    next = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(next)
    next = tf.keras.layers.Flatten()(next)
    embedding_out_layer = tf.keras.layers.Dense(EMB_SIZE)(next)
    model = tf.keras.Model(inputs=input_img, outputs=embedding_out_layer)
    model.summary()
    
    if L2NORM:
        embedding_model = tf.keras.Model(inputs=model.layers[0].input, outputs=tf.keras.layers.Lambda(lambda temp: tf.keras.backend.l2_normalize(temp,axis=1))(embedding_out_layer))
    else:
        embedding_model = tf.keras.Model(inputs=model.layers[0].input, outputs=embedding_out_layer)
    embedding_model.summary()
    
    # build the Siamese network
    input_anchor = tf.keras.layers.Input(shape=(tilesize,tilesize,1))
    input_positive = tf.keras.layers.Input(shape=(tilesize,tilesize,1))
    input_negative = tf.keras.layers.Input(shape=(tilesize,tilesize,1))
    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)
    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
    net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
    net.summary()
    net.compile(loss=triplet_loss, optimizer='adam')


# ## Optionally load from a checkpoint, for example to resume training, or from another pre-trained model

# In[ ]:


embedding_model.load_weights('asil20_weights.hdf5')
#net.load_weights('checkpoints/weights-0120.hdf5')


# ## Train network, or load pre-trained model

# In[ ]:


if TRAIN_MODE:
    steps_per_epoch = int(x_train.shape[0]/BATCH_SIZE)
    if x_valid.shape[0]==0:  # data set contains no validation data
        validation_steps = None
        validation_data = None
    else:                    # data set DOES contain validation data, set appropr
        validation_steps = int(x_valid.shape[0]/BATCH_SIZE)
        validation_data=data_generator(x_valid,y_valid,valid_tiles_per_image,BATCH_SIZE)
    history = net.fit(
        data_generator(x_train,y_train,tiles_per_image,BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=[Metrics((x_valid, y_valid),valid_tiles_per_image,embedding_model), 
                   Plotter(),
                   tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/weights-{epoch:04d}.hdf5', save_weights_only=True, save_best_only=True, monitor='val_loss')],
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data, validation_steps=validation_steps
    )
else:
    embedding_model.load_weights(MODEL_FILENAME+'.hdf5')


# ## Optionally save model

# In[ ]:


#embedding_model.save_weights(filepath='saved_model.hdf5')


# ## Compute training set tile features on final/trained network. Use tile features to compute training set image distances using two methods. Report mean self-similarity rank on training set (lower is better)

# In[ ]:


embs = embedding_model.predict(x_train)
[img_distances1, _, _] = compute_image_distances(embs, tiles_per_image)
print('mean self-similarity rank: '+str("{:.2e}".format(mean_self_similarity_rank(img_distances1))))


# <hr style="height:3px;border:none;background-color:#DC143C" />
# 
# 
# ## Load the test set, report results including comparison with prior approaches.

# In[ ]:


# import test tile data
npzfile_test = np.load(TESTDATA_FILE)
x_test = npzfile_test['x_test']
y_test = npzfile_test['y_test']
fnames_test = npzfile_test['fnames_test']          # filenames corresponding to every tile in x_test (not used below, but helpful for debug)
classnames_test = npzfile_test['classnames_test']  # ordered image names (which subsequently appear in img_distanceX matrices, helpful for debug)
test_tiles_per_image = x_test.shape[0]//classnames_test.shape[0]  # number of tiles per image (assumed adjacent in test set)

# compute image distances on test set
embs_test = embedding_model.predict(x_test)
[_, img_distances2_test, centroids] = compute_image_distances(embs_test, test_tiles_per_image)


# ## Save test set distances and features to Matlab file

# In[ ]:


# Save these variables to Matlab:
#     D -- distance matrix ( N x N )
#     f -- feature vector ( N x EMB_SIZE )
#     fnames -- cell array string of filenames ( N x 1 )

# permutation from alphabetical ordering to Yale IPCH ordering
permut = [344, 345, 346, 347, 12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 52, 53, 54, 55, 48, 
          49, 50, 51, 24, 25, 26, 27, 28, 29, 30, 31, 20, 21, 22, 23, 16, 17, 18, 19, 44, 45, 46, 47, 
          40, 41, 42, 43, 32, 33, 34, 35, 36, 37, 38, 39, 84, 85, 86, 87, 80, 81, 82, 83, 72, 73, 74, 
          75, 76, 77, 78, 79, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 124, 125, 
          126, 127, 120, 121, 122, 123, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 
          103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 148, 149, 
          150, 151, 144, 145, 146, 147, 140, 141, 142, 143, 128, 129, 130, 131, 132, 133, 134, 135, 136, 
          137, 138, 139, 172, 173, 174, 175, 168, 169, 170, 171, 164, 165, 166, 167, 156, 157, 158, 159, 
          160, 161, 162, 163, 192, 193, 194, 195, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 
          187, 188, 189, 190, 191, 200, 201, 202, 203, 204, 205, 206, 207, 196, 197, 198, 199, 208, 209, 
          210, 211, 212, 213, 214, 215, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 228, 
          229, 230, 231, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 256, 257, 258, 259, 
          252, 253, 254, 255, 248, 249, 250, 251, 244, 245, 246, 247, 284, 285, 286, 287, 288, 289, 290, 
          291, 292, 293, 294, 295, 280, 281, 282, 283, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 
          270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 340, 341, 342, 343, 332, 333, 334, 335, 336, 
          337, 338, 339, 328, 329, 330, 331, 324, 325, 326, 327, 296, 297, 298, 299, 300, 301, 302, 303, 
          304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 
          323, 524, 525, 526, 527, 516, 517, 518, 519, 520, 521, 522, 523, 604, 605, 606, 607, 608, 609, 
          610, 611, 596, 597, 598, 599, 600, 601, 602, 603, 592, 593, 594, 595, 588, 589, 590, 591, 584, 
          585, 586, 587, 580, 581, 582, 583, 620, 621, 622, 623, 612, 613, 614, 615, 616, 617, 618, 619, 
          636, 637, 638, 639, 632, 633, 634, 635, 624, 625, 626, 627, 628, 629, 630, 631, 356, 357, 358, 
          359, 348, 349, 350, 351, 352, 353, 354, 355, 652, 653, 654, 655, 648, 649, 650, 651, 640, 641, 
          642, 643, 644, 645, 646, 647, 372, 373, 374, 375, 368, 369, 370, 371, 360, 361, 362, 363, 364, 
          365, 366, 367, 388, 389, 390, 391, 376, 377, 378, 379, 384, 385, 386, 387, 400, 401, 402, 403, 
          392, 393, 394, 395, 380, 381, 382, 383, 396, 397, 398, 399, 420, 421, 422, 423, 404, 405, 406, 
          407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 440, 441, 442, 443, 424, 425, 
          426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 452, 453, 454, 455, 448, 
          449, 450, 451, 444, 445, 446, 447, 540, 541, 542, 543, 528, 529, 530, 531, 532, 533, 534, 535, 
          536, 537, 538, 539, 560, 561, 562, 563, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 
          555, 556, 557, 558, 559, 700, 701, 702, 703, 696, 697, 698, 699, 692, 693, 694, 695, 688, 689, 
          690, 691, 716, 717, 718, 719, 712, 713, 714, 715, 704, 705, 706, 707, 708, 709, 710, 711, 732, 
          733, 734, 735, 728, 729, 730, 731, 724, 725, 726, 727, 720, 721, 722, 723, 744, 745, 746, 747, 
          748, 749, 750, 751, 740, 741, 742, 743, 736, 737, 738, 739, 764, 765, 766, 767, 760, 761, 762, 
          763, 752, 753, 754, 755, 756, 757, 758, 759, 780, 781, 782, 783, 776, 777, 778, 779, 768, 769, 
          770, 771, 772, 773, 774, 775, 796, 797, 798, 799, 792, 793, 794, 795, 788, 789, 790, 791, 784, 
          785, 786, 787, 812, 813, 814, 815, 808, 809, 810, 811, 804, 805, 806, 807, 800, 801, 802, 803, 
          828, 829, 830, 831, 824, 825, 826, 827, 820, 821, 822, 823, 816, 817, 818, 819, 844, 845, 846, 
          847, 840, 841, 842, 843, 832, 833, 834, 835, 836, 837, 838, 839, 856, 857, 858, 859, 848, 849, 
          850, 851, 852, 853, 854, 855, 460, 461, 462, 463, 456, 457, 458, 459, 464, 465, 466, 467, 468, 
          469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 492, 493, 494, 495, 480, 481, 482, 483,
          484, 485, 486, 487, 488, 489, 490, 491, 512, 513, 514, 515, 504, 505, 506, 507, 508, 509, 510, 
          511, 496, 497, 498, 499, 500, 501, 502, 503, 668, 669, 670, 671, 656, 657, 658, 659, 664, 665, 
          666, 667, 660, 661, 662, 663, 680, 681, 682, 683, 684, 685, 686, 687, 672, 673, 674, 675, 676, 
          677, 678, 679, 152, 153, 154, 155, 576, 577, 578, 579, 572, 573, 574, 575, 564, 565, 566, 567, 
          568, 569, 570, 571]

sio.savemat('output.mat', {'D':img_distances2_test[permut] [:,permut], 
                           'fnames':np.asarray(classnames_test[permut], dtype='object'), 
                           'f':centroids[permut]}, oned_as='column')

