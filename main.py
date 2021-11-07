import cv2 as cv
import numpy as np
from display_helper import imshow
from tqdm import tqdm
import sys


def calculate_gradient(image, axis='x'):
    """
    Calculates gradiant of an image in the given dimension.
    I have added an extra parameter to take the direct along whihc grad has to be calculated
    This contains implementation of both custom sobel and usage of cv2's sobel
    I have commented out custom implemtation entry cause it takes ~30 mins for execution
    :param image: input image
    :param axis: value can be 'x' or 'y'
    :return: gradient of image
    """

    def apply_custom_filter(image, width, pixel_wise_filter):
        """
        This is a handy method I have implemented to cut short most of the filtering & convolving operations.
        The method just requires a kernel operation lambda/method as param and uses it compute resultant
        :param image: image
        :param width: widow width of the kernel on which operation happens
        :param pixel_wise_filter: a method that computes a value for output image by taking in the sub-image overlapping on kernel
        :return: kernel operation output
        """
        new_im = np.zeros(image.shape)
        padding = width // 2
        # add padding
        img_pd = cv.copyMakeBorder(image, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=0)
        for i in range(padding, img_pd.shape[0] - padding):
            for j in range(padding, img_pd.shape[1] - padding):
                # for every sub image window, compute output and set it in new image
                new_im[i - padding, j - padding] = pixel_wise_filter(
                    img_pd[i - padding:i + padding + 1, j - padding:j + padding + 1])
        return new_im

    def sobel_kernel_x(sub_img):
        """
        this method is used for computing the sobel X gradient kernel convolution output
        :param sub_img: a patch of image
        :return: kernel output
        """
        kernel = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        return (kernel * sub_img).sum()

    def sobel_kernel_y(sub_img):
        """
        this method is used for computing the sobel Y gradient kernel convolution output
        :param sub_img: a patch of image
        :return: kernel output
        """
        kernel = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        return (kernel * sub_img).sum()

    # uncomment this line to use custom sobel gradient (but this is a bit slower compared to the open CV's sobel)
    # grad = apply_custom_filter(image, 3, (sobel_kernel_x if axis=='x' else sobel_kernel_y))
    if axis == 'x':
        grad = cv.Sobel(image, cv.CV_32F, 1, 0)
    else:
        grad = cv.Sobel(image, cv.CV_32F, 0, 1)
    return grad


def get_energy_map(image):
    """
    calculate the energy matrix of the image
    energy is abs(gradX) + abs(gradY) , as defined in the paper referenced above
    :param image: input image
    :return: energy matrix, same shape as the input image
    """
    # convert image to gray
    img_new = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # energy = abs(gradX) + abs(gradY)
    energy = np.abs(calculate_gradient(img_new, 'x')) + np.abs(calculate_gradient(img_new, 'y'))
    # normalize energy
    # energy = energy / energy.max()
    return energy


def calculate_cumulative_min_map(energy):
    cum_min_map = np.zeros(energy.shape)

    def get_pixel(i, j):
        """
        a helper method to safely fetch a pixel from cal_cumulative_min_map
        """
        if (energy.shape[1] > j >= 0):
            return cum_min_map[i, j]
        else:
            return np.PINF

    for p in range(energy.shape[0]):
        for q in range(energy.shape[1]):
            # recursive DP logic for finding optimal path
            cum_min_map[p, q] = energy[p, q] + min(get_pixel(p - 1, q - 1), get_pixel(p - 1, q),
                                                   get_pixel(p - 1, q + 1))
    return cum_min_map


def delete_seam(cum_min_map, image):
    """
    This function is used to remove one single seam
    :param cal_cumulative_min_map:
    :param image:
    :return:
    """

    def get_pixel(i, j):
        """
        a helper method to safely fetch a pixel from cal_cumulative_min_map
        """
        if (cum_min_map.shape[1] > j >= 0):
            return cum_min_map[i, j]
        else:
            return np.PINF

    seam_len = image.shape[0]
    # seam initialization
    seam = np.zeros((seam_len), dtype=np.uint32)
    # set last index of seam
    seam[seam_len - 1] = np.argmin(cum_min_map[seam_len - 1, :])
    # finidng best seam with backtracking
    for i in np.arange(seam_len - 2, -1, -1):
        t = np.argmin([get_pixel(i, seam[i + 1] - 1), get_pixel(i, seam[i + 1]), get_pixel(i, seam[i + 1] + 1)])
        seam[i] = seam[i + 1] + (t - 1)
    new_image = np.zeros((image.shape[0], image.shape[1] - 1, image.shape[2]), dtype=np.uint8)
    # stitching the two parts with the seam removed
    for t in range(seam_len):
        new_image[t, :seam[t]] = image[t, :seam[t]]
        new_image[t, seam[t]:] = image[t, seam[t] + 1:]
    # below lines if uncommented highlights the seam
    imcpy = np.copy(image)
    for t in range(seam_len):
        imcpy[t, seam[t], :]=[0,0,0]
    # imshow(imcpy)
    # cv.imwrite("results/"+str(image.shape[1])+".png", imcpy)
    return new_image


def seam_carve(img, n=25):
    """
    removes n seams of the input image and returns final and intermediate
     images (1 per each seam removed)
    :param img: input image
    :return: a tuple of final image and intermediate image list
    """
    new_img = img.copy()
    inter_images = [img]
    for i in tqdm(range(n)):
        energy_map = get_energy_map(new_img)
        cum_min_map = calculate_cumulative_min_map(energy_map)
        new_img = delete_seam(cum_min_map, new_img)
        # cv.imwrite("results/"+str(i)+".png", new_img)
        inter_images.append(new_img)
    return new_img, inter_images


if __name__ == '__main__':
    img = cv.imread('image.jpg' if len(sys.argv) < 2 else sys.argv[1])[:, :, [2, 1, 0]]
    n = img.shape[1] // 3
    new_img, intermediate_images = seam_carve(img, n)
    slider_attr = [{'label': 'Vertical Seams removed', 'valmin': 0, 'valmax': n - 1, 'valint': 1}]

    def update_img(x, axs, sliders, buttons):
        return [0], [intermediate_images[int(sliders[0].val)]]
    imshow(new_img, slider_attr=slider_attr, slider_callback=[update_img])
