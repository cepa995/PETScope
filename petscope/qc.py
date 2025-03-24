import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def convert_to_uint8(image):
    """
    Normalizes image to UINT8, because OpenCV works with specific
    data types when it comes to certain functionalities such as
    .addWeighted, etc.

    :param image - numpy array which has to be converted to UINT8
    :returns normalized numpy array with UINT8 data type
    """
    # Normalize image to range [0, 255] and convert to uint8
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image.astype(np.uint8)

def get_reconstruction_planes(image_3d):
    """
    Get reconstruction planes from 3D medical image

    :param image_3d - numpy array of a 3D medical image
    :returns sagittal, coronal and axial planes
    """
    assert image_3d.ndim == 3
    n_i, n_j, n_k = image_3d.shape
    # saggital
    center_i1 = int((n_i - 1) / 2)
    # transverse
    center_j1 = int((n_j - 1) / 2)
    # axial slice
    center_k1 = int((n_k - 1) / 2)
    # Get SAGITTAL plane
    sagittal = image_3d[center_i1, :, :]
    coronal  = image_3d[:, center_j1, :]
    axial    = image_3d[:, :, center_k1]
    return sagittal, coronal, axial

def show_slices(slices, figsize=(15, 5), cmap='gray'):
    """
    Function to display a row of image slices
    :param slices - a list of numpy 2D image slices
    """
    fig, axes = plt.subplots(1, len(slices), figsize=figsize)
    for i, slice in enumerate(slices):
        axes[i].imshow(slice, origin="lower")


if __name__ == "__main__":
    # Load your image (replace 'path_to_image' with your image file path)
    image = nib.load('C:/Users/StefanCepa995/Desktop/rpet_4d.nii')
    image_data = image.get_fdata()
    sagittal, coronal, axial = get_reconstruction_planes(image_data[:, :, :, 11])
    
    # Create a window and specify the dimensions
    cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image Window', 1200, 1000)  # Width = 800, Height = 600

    # Display the image in a window
    cv2.imshow('Image Window', sagittal)

    # Wait indefinitely until a specific key is pressed (e.g., 'q' to quit)
    while True:
        key = cv2.waitKey(0) & 0xFF  # Waits indefinitely and gets the key code
        if key == ord('q'):  # Replace 'q' with any other key you want to use
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()