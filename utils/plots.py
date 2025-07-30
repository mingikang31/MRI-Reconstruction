import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_image_with_zoom(image, roi, border_width=0.5, zoom_size="35%"):
    """
    Displays an image with a zoomed-in inset anchored exactly to the bottom-right corner.
    
    Args:
        image (np.ndarray): The full image to display.
        roi (tuple): The region of interest (x_start, y_start, width, height).
        border_width (float, optional): The line width for the red borders. Defaults to 0.5.
        zoom_size (str, optional): The size of the zoom box as a percentage
                                 of the main image's size. Defaults to "40%".
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Display the full image
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    
    # 2. Draw the ROI rectangle on the full image
    x, y, w, h = roi
    ax.add_patch(
        Rectangle((x, y), w, h,
                 edgecolor='red',
                 facecolor='none',
                 linewidth=border_width)
    )
    
    # 3. Create an inset axis anchored exactly to the bottom-right corner
    ax_zoom = inset_axes(
        ax,
        width=zoom_size,
        height=zoom_size,
        loc='lower right',
        bbox_to_anchor=(164, 16, 450, 450),  # Adjust the bbox_to_anchor for exact corner alignment

        borderpad=0  # Set to 0 for exact alignment with corner
    )
    
    # 4. Crop the image and display it on the inset axis
    cropped_image = image[y:y+h, x:x+w]
    ax_zoom.imshow(cropped_image, cmap='gray')
    
    # 5. Style the inset axis
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(border_width)
    
    plt.tight_layout()  # This can help with alignment
    plt.show()

# Example Usage
# Load your image in grayscale
image_path = "/export1/project/mingi/qmri/results/Full/RUnet_Full_Dataset_GradientLoss_Real/echo_2/out/img_50.png"
your_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the Region of Interest for your 234x176 image
# Format is (x_start, y_start, width, height)
roi = (35, 60, 55, 55)

# Call the function with your desired border width 
plot_image_with_zoom(your_image, roi, border_width=1)