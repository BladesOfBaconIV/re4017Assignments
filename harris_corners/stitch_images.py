# Assignment 2: Stitching images together
# Authors:
#   Darragh Glavin:     16189183
#   Lorcan Williamson:  16160703
#   Yilin Mou:          18111602
# Python 3.7

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

IMAGE_NAME = "arch"
IMAGES = [f'./images/{IMAGE_NAME}1.png', f'./images/{IMAGE_NAME}2.png']
EPSILON = 1e-10                 # Epsilon value used to avoid division by 0

SHOW_INTERMEDIATE = False        # Show intermediate steps


def plot_return(flag, title='', background_image=False, as_points=False, transpose=False, **plot_args):
    """
    Decorator function to allow easy plotting of intermediate results
    :param flag: flag variable to use (If flag == True plot)
    :param title: Title for the plot
    :param background_image: Whether to use a background image, if True will use the first arg of func as the background
    :param as_points: Whether to plot the result of func as (x, y) points, assumes they are (r, c)
    :param transpose: Transpose the image before plotting
    :param plot_args: extra kwargs to pass to plt.imshow or plt.scatter (Depending on as_points)
    :return: decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if flag:
                for r in result if type(result) == tuple else [result]:
                    plt.figure()
                    if background_image:
                        plt.imshow(*args, cmap='gray')
                    if as_points:
                        plt.scatter(*zip(*[(x, y) for y, x in result]), **plot_args)
                    else:
                        plt.imshow(r if not transpose else r.T, **plot_args)
                    plt.title(title)
            return result
        return wrapper
    return decorator


@plot_return(SHOW_INTERMEDIATE, title='Edge detection', cmap='gray')
def find_edges(image, sigma=0.5):
    """
    Find the x, y (column, row) edges of an image using gaussian derivative kernel
    :param image: np.array of pixel values (Should be grayscale)
    :param sigma: sigma value for gaussian kernel (Default 0.5)
    :return: (Ix, Iy) image edges in x and y direction
    """
    # Needed as image (uint8) does not have the precision to store intermediate values of convolution
    Ix, Iy = np.zeros(image.shape), np.zeros(image.shape)
    gaussian_filter(image, sigma=sigma, order=(0, 1), output=Ix)
    gaussian_filter(image, sigma=sigma, order=(1, 0), output=Iy)
    return Ix, Iy


def structure_tensor_values(Ix, Iy, sigma):
    """
    Calculate the A, B, C values of the local structure tensor, using a Gaussian kernel function
    :param Ix: Edges in X direction (Columns)
    :param Iy: Edges in Y direction (Rows)
    :param sigma: sigma for the Gaussian filter
    :return: (A, B, C) matrices corresponding to (w*Ix^2, w*(Ix.Iy), w*Iy^2)
    """
    return (
        gaussian_filter(Ix * Ix, sigma=sigma),    # A
        gaussian_filter(Ix * Iy, sigma=sigma),    # B
        gaussian_filter(Iy * Iy, sigma=sigma)     # C
    )


@plot_return(SHOW_INTERMEDIATE, title='Harris Response', cmap='hot')
def harris_response(image, sigma=1):
    """
    Get the Harris response of an image
    :param image: greyscale image
    :param sigma: sigma for edge detection
    :return: Harris response
    """
    Ix, Iy = find_edges(image, sigma=sigma)  # Edges in x and y direction
    A, B, C = structure_tensor_values(Ix, Iy, sigma=2.5*sigma)
    return ((A * C) - (B ** 2)) / (A + C + EPSILON)  # Harris response


@plot_return(SHOW_INTERMEDIATE, title='Harris Interest Points', background_image=True, as_points=True, s=6, c='r')
def find_harris_interest_points(image, threshold=0.1, sigma=1, suppression_size=10):
    """
    Find the Harris interest points of an image, thresholded using Szeliski harmonic mean
    :param image: np.array of pixel values (Should be grayscale)
    :param threshold: Proportion max interest point value to threshold at (Default 0.1)
    :param sigma: Sigma for edge detection filter (Default 1)
    :param suppression_size: number of pixels in the x and y direction (+/-)
                                to non-max suppress around each interest point (Default 10)
    :return: np.array of interest points ([[r1, c1], [r2, c2], ...])
    """
    R = harris_response(image, sigma=sigma)
    R_th = R > (R.max() * threshold)                        # Thresholded Harris response
    interest_points = np.array(R_th.nonzero()).T            # Co-ordinates of interest points (r, c)
    values = [R[c[0], c[1]] for c in interest_points]       # Values of response at interest points
    value_order = np.argsort(values)
    allowed_loc = np.zeros(image.shape, dtype='bool')
    allowed_loc[suppression_size:-suppression_size, suppression_size:-suppression_size] = True  # Not allowed near edge
    best_points = []
    for r, c in interest_points[value_order[::-1]]:
        if allowed_loc[r, c]:
            best_points.append((r, c))
            allowed_loc[r-suppression_size:r+suppression_size+1, c-suppression_size:c+suppression_size+1] = False
    return np.array(best_points)


@plot_return(SHOW_INTERMEDIATE, title='Patch Descriptors', transpose=True, cmap='copper')
def make_image_patch_descriptors(image, hips, patch_size=11):
    """
    Create a matrix of flattened patches taken from around the Harris interest points
    :param image: Image to take patches of
    :param hips: Harris interest points of the image (r, c)
    :param patch_size: Size of the patches (Default 11)
    :return: np.array of patch descriptors
    """
    patch_descriptors = []
    delta = (patch_size - 1) // 2
    for r, c in hips:
        patch = np.ravel(image[r-delta : r+delta+1, c-delta : c+delta+1])   # 1d view of patch
        patch_descriptors.append(patch / np.linalg.norm(patch))             # normalise each patch
    return np.array(patch_descriptors)


def calculate_image_translations(image1, hips1, image2, hips2, threshold=0.95):
    """
    Calculate the translation vectors based on Harris interest points of the two images
    :param image1: np.array grayscale image 1
    :param hips1: Harris interest points of image 1
    :param image2: np.array grayscale image 2
    :param hips2: Harris interest points of image 2
    :param threshold: Threshold for response to be considered a good match (Default 0.95)
    :return: translation vector of image 2 relative to image 1
    """
    M1 = make_image_patch_descriptors(image1, hips1)
    M2 = make_image_patch_descriptors(image2, hips2)
    response_matrix = np.dot(M1, M2.T)
    response_matrix = response_matrix > threshold
    match_coords = np.array(response_matrix.nonzero())
    return hips1[match_coords[0, :]] - hips2[match_coords[1, :]]


def exhaustive_ransac(translations, max_delta=1.6):
    """
    Return the option that the majority of options agree with
    :param translations: List of translation vectors
    :param max_delta: max difference between vector end points to gain a vote (in pixels)
    :return: The most agreeable option
    """
    votes = np.zeros((len(translations)))
    for i, t in enumerate(translations):
        deltas = np.linalg.norm((translations - t), axis=1)
        votes[i] = (deltas < max_delta).sum()
    return translations[votes.argmax()]


def stitch_images(im1: Image, im2: Image):
    """
    Stitch two RGB images together using Harris interest points
    :param im1: Image 1
    :param im2: Image 2
    :return: Image 1 stitched with image 2
    """
    im1_gray, im2_gray = np.array(im1.convert('L')), np.array(im2.convert('L'))
    hips1 = find_harris_interest_points(im1_gray)
    hips2 = find_harris_interest_points(im2_gray)
    possible_translations = calculate_image_translations(im1_gray, hips1, im2_gray, hips2)
    assert len(possible_translations), "No matches found for Harris interest points"
    dy, dx = exhaustive_ransac(possible_translations)  # Returns (row, column), need to change to (x, y)
    size, im1_offset, im2_offset = {
        (True, True):   ((max(im2.size[0]+dx, im1.size[0]), max(im2.size[1]+dy, im1.size[1])), (0, 0), (dx, dy)),    # +dx, +dy
        (True, False):  ((max(im2.size[0]+dx, im1.size[0]), max(im1.size[1]-dy, im2.size[1])), (0, -dy), (dx, 0)),   # +dx, -dy
        (False, True):  ((max(im1.size[0]-dx, im2.size[0]), max(im2.size[1]+dy, im1.size[1])), (-dx, 0), (0, dy)),   # -dx, +dy
        (False, False): ((max(im1.size[0]-dx, im2.size[0]), max(im1.size[1]-dy, im2.size[1])), (-dx, -dy), (0, 0))   # -dx, -dy
    }[(dx > 0, dy > 0)]
    combined_images = Image.new(im1.mode, size)
    combined_images.paste(im1, im1_offset)
    combined_images.paste(im2, im2_offset)
    return combined_images


def test_rotation_effect(im1: Image, im2: Image, start=1, stop=10, step=2):
    """
    Tests the affect small rotations of one image has on the ability to align images
    :param im1: First to be aligned
    :param im2: Second to be aligned (This is the image that will be rotated)
    :param start: Angle to start testing at (int)
    :param stop: Angle to stop testing at (inclusive) (int)
    :param step: Steps between test angles
    """
    for theta in range(start, stop+1, step):
        try:
            im2_rotated = im2.rotate(theta)
            combined = stitch_images(im1, im2_rotated)
            plt.figure()
            plt.imshow(combined)
            plt.title(f'Image 2 rotated {theta} degrees')
        except AssertionError as e:
            print(f"Failed for rotation {theta} degrees: {e}")


def test_scaling_affect(im1: Image, im2: Image, factor=2, num_steps=10):
    """
    Test the effect scaling  of the images has on the ability to align them
    :param im1: First to be aligned
    :param im2: Second to be aligned (This is the image that will be scaled)
    :param factor: Image will be scaled between [1/factor, ..., factor]
    :param num_steps: Number of steps between min and max scaling factor
    """
    step = (factor - 1/factor)/num_steps
    for f in np.arange(1/factor, factor+step, step):
        try:
            im2_scaled = im2.resize((int(im2.width * f), int(im2.height * f)))
            combined = stitch_images(im1, im2_scaled)
            plt.figure()
            plt.imshow(combined)
            plt.title(f'Image 2 scaled {100*f:.0f}%')
            plt.savefig(f'./report/images/{IMAGE_NAME}-scaled-{100*f:.0f}.png')
        except AssertionError as e:
            print(f"Failed for scale {100*f:.0f}%: {e}")


if __name__ == "__main__":
    images = [Image.open(path) for path in IMAGES]
    final_image = stitch_images(*images)
    plt.figure()
    plt.imshow(final_image)
    plt.show()