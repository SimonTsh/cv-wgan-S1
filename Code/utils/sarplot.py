import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
# Need to install scikit-image to use the following modules
from skimage import img_as_float


def plot_amp_img(cplx_image):
    plt.figure()
    plt.imshow(20 * np.log10(np.abs(cplx_image) + 1e-2), cmap=plt.cm.gray,
               aspect=cplx_image.shape[1] / cplx_image.shape[0])
    plt.colorbar()
    plt.show()


def plot_equalized_img(data, method="equal", crop=None, bins=256, savefig=False):
    """ A method to plot single UAVSAR SLC 1x1 data
        Inputs:
            * crop = [lowerIndex axis 0, UpperIndex axis 0, lowerIndex axis 1, UpperIndex axis 1], a list of int, to read a portion of the image, it reads the whole image if None.
            * method = "stretch" or "equal", based from histogram equalization, see https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html

    """

    for i, d in enumerate(data):
        if crop is not None:

            img = np.log10(np.abs(d[crop[0]:crop[1], crop[2]:crop[3]]) + 1e-8)

        else:

            img = np.log10(np.abs(d) + 1e-2)

        img = (img - img.min()) / (img.max() - img.min())  # rescale between 0 and 1

        if method == "stretch":
            p2, p98 = np.percentile(img, (2, 98))
            img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

        elif method == "equal":
            img_rescale = exposure.equalize_hist(img)

        else:
            raise NameError("wrong 'method' or not defined")

        fig = plt.figure(figsize=(8, 8))
        image = img_as_float(img_rescale)
        plt.imshow(image, cmap=plt.cm.gray)  # aspect = img.shape[1]/img.shape[0])
        if savefig:
            plt.savefig('SAR_' + "{}.png".format(i))

    plt.show()


