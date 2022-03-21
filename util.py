

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from skimage import measure
import nibabel as nib

def lung_area(lung_mask, vox):


    'to calculate area of lung per slice multiply the number of lung pixels per slice with the pixel dimensions'

    num_pixels_lung = np.sum(lung_mask)
    area_lung = num_pixels_lung * vox[0] * vox[1] * vox[2]

    return area_lung




def create_mask(image, lung_contour):
    lung_mask = np.array(Image.new('L', image.shape, 0))
    for contour in lung_contour:
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', image.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        lung_mask += mask

    return lung_mask





def length_contours(contour):
    """check the length of the contours"""

    delta_y = contour[0, 1] - contour[-1, 1]
    delta_x = contour[0, 0] - contour[-1, 0]
    return np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))


def closed_contour(contour):
    "check if the contour is closed"
    if length_contours(contour) < 1:
        return True
    else:
        return False


def lungs_cont(contours):
    """
    if the contours passes the checks i.e. is a closed volume of a specific size keep it and discard the other
    with trial and error the chosen volume threshold was 15000 for the given dataset. 3 possible scenarios arise.
    1. the 2 lung regions are not separated and make one large contour, and the body contour is not closed leaving us
    with 1 contour that passes the check
    2. the 2 lungs are separated and the body in not a closed contour leaving us with 2 contours that pass the initial
    check
    3. the lungs are separated and the body is a closed contour giving us more than 2 contours. In this case assuming
    that the body contour is the largest, it is removed leaving behind the 2 lung contours.
    4. A possible fourth scenario is single lung contour and tthe bost contour is closed but scenario did not come up
    for the provided dataset.

    """
    lung_contours = []
    vol = []

    for contour in contours:
        conv_hull = ConvexHull(contour)

        if conv_hull.volume > 15000 and closed_contour(contour):
            lung_contours.append(contour)
            vol.append(conv_hull.volume)
        #elif conv_hull.volume <= 15000 and closed_contour(contour):

    if len(lung_contours) == 1:
        return lung_contours
    elif len(lung_contours) == 2:
        return lung_contours
    elif len(lung_contours) > 2:
        vol, lung_contours = (list(t) for t in
                              zip(*sorted(zip(vol, lung_contours))))
        lung_contours.pop(-1)
        return lung_contours
    


def sep_main_vessels(lung_mask, image, vox):
    vessels = lung_mask * image
    vessels[vessels == 0] = -1000
    vessels[vessels >= -500] = 1
    vessels[vessels < -500] = 0
    #plt.figure()
    #plt.imshow(vessels)
    #plt.show()
    vessel_vol_slice = np.sum(vessels) * vox[0] * vox[1] * vox[2]

    return vessels, vessel_vol_slice

def binarize(image, upper_threshold, lower_threshold):
    image = image.clip(lower_threshold, upper_threshold)

    image[image != upper_threshold] = 1
    image[image == upper_threshold] = 0



    return image




def create_contour(image_bin, image):

    contours = measure.find_contours(image_bin, 0.8)
    #fig, ax = plt.subplots()
    #ax.imshow(image)

    #for contour in contours:
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    #plt.show()

    return contours


def save_nifty(lung_mask, name, affine):

    #save mask in nifty format

    lung_mask[lung_mask == 1] = 255
    if isinstance(lung_mask, list):
        ni_img = nib.Nifti1Image(lung_mask,affine)
    else:
        ni_img = nib.Nifti1Image(lung_mask, affine)

    nib.save(ni_img, name + '.nii.gz')


