"""
Usage: python util_inference_example.py image annotations
Adapted from the dense_inference.py to demonstate the usage of the util
functions.
"""

import sys
import numpy as np
import cv2
import densecrf as dcrf
import matplotlib.pylab as plt
from skimage.segmentation import relabel_sequential

from utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian

fn_im = 'im2.ppm' #sys.argv[1]
fn_anno = 'anno2.ppm' #sys.argv[2]

NIT = 5

def print_params(crf):
    print 'Unary Parameters:', np.array(crf.unaryParameters)
    print 'Label compatability params:', np.array(
            crf.labelCompatibilityParameters)
    print 'Kernel Parameters:', np.array(crf.kernelParameters)

##################################
### Read images and annotation ###
##################################
img = cv2.imread(fn_im)
labels = relabel_sequential(cv2.imread(fn_anno, 0))[0].flatten()
M = 4 #labels.max() + 1

###########################
### Setup the CRF model ###
###########################
crf = dcrf.DenseCRF(img.shape[0] * img.shape[1], M)

# Get logistic unary features
im_feat = np.rollaxis(img, 2).astype(np.float32) / 255.
logistic_feature = np.ones((4, len(labels)), np.float32)
logistic_feature[:3, :] = im_feat.reshape([3, -1])
#logistic_feature[[2,1,0], :] = im_feat.reshape([3, -1])

logistic_transform = np.random.uniform(-0.01, 0.01, (M, 4)).astype(np.float32)
crf.setUnaryEnergy(logistic_transform, logistic_feature)
#crf.unaryParameters = np.array([-0.00680375 ,  0.00211234 , -0.00566198  ,
# -0.0059688 , -0.00823295 ,  0.00604897 ,  0.00329555,  -0.00536459,   0.00444451 ,  -0.0010794 , 0.000452059,  -0.00257742 ,  0.00270431, -0.000268018 , -0.00904459 ,  -0.0083239], dtype=np.float32)

# This creates the color-independent features and then add them to the CRF
feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
crf.addPairwiseEnergy(feats, compat=1,
                      kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

# This creates the color-dependent features and then add them to the CRF
feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                  img=img, chdim=2)
crf.addPairwiseEnergy(feats, compat=np.identity(M, np.float32),
                      kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

####################################
### Do inference and compute map ###
####################################
Q = crf.inference(NIT)
map = np.argmax(Q, axis=0).reshape(img.shape[:2])
res1 = map.astype('float32') * 255 / map.max()
print_params(crf)

############################
### Learn the parameters ###
############################
obj = dcrf.IoUObjective(labels.astype(np.int32))

print 'Initial parameters'
print print_params(crf)

for u, p, k, text in [(True, False, False, 'unary'),
                      (True, True, False, 'unary and pairwise'),
                      (True, True, True, 'full CRF')]:
    print '\nOptimize {}'.format(text)
    energy = dcrf.CRFEnergy(crf, obj, niter=NIT, unary=u, pairwise=p, kernel=k)
    energy.setL2Norm(0.001)
    p = energy.learn_parameters(2, True)

Q = crf.inference(NIT)
map = np.argmax(Q, axis=0).reshape(img.shape[:2])
res2 = map.astype('float32') * 255 / map.max()

########################
### Plot all results ###
########################
fig, ax = plt.subplots(1, 4, figsize=[20, 5])
ax[0].imshow(img)
ax[0].set_title('Original input image', fontsize=15)
ax[1].imshow(labels.reshape(img.shape[:2]))
ax[1].set_title('Original annotation', fontsize=15)
ax[2].imshow(res1)
ax[2].set_title('CRF inference only', fontsize=15)
ax[3].imshow(res2)
ax[3].set_title('CRF after optimization', fontsize=15)

plt.show()
