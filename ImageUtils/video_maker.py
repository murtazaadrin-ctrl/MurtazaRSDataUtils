import os
import numpy as np
import tempfile
import cv2
import matplotlib.pyplot as plt

def numpy_array_to_image(array, filename, text):
    """Convert a NumPy array to an image and save it as a file."""
    plt.imshow(array, cmap='gray')  # Use 'gray' for grayscale images, adjust cmap as needed
    plt.axis('off')  # Hide axes
    plt.text(0.90, 0.1, text, color='red', fontsize=16,ha='right', va='bottom', transform=plt.gca().transAxes)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    #if text:
        # Add text to the image

    plt.close()

def create_video_from_frames(frames, texts , output_filename, fps=24):
    """Create a video from a list of NumPy array frames."""
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory for saving images
    image_files = []

    # Convert each frame to an image file
    for i, frame in enumerate(frames):
        image_filename = os.path.join(temp_dir, f'frame_{i:04d}.png')

        b= frame
        b[np.isnan(b)]=0
        if np.mean(b) !=0:
            numpy_array_to_image(frame, image_filename , texts[i])
            #print(texs[i])
            image_files.append(image_filename)

    #fps=5

    temp = cv2.imread(image_files[0])
    height, width = temp.shape[:2]




    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 files
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        out.write(img)

    # Create video clip from images
    # clip = ImageSequenceClip(image_files, fps=fps)
    # clip.write_videofile(output_filename, codec='libx264')

    # Clean up temporary files
    for image_file in image_files:
        os.remove(image_file)
    os.rmdir(temp_dir)


def video_maker(
    sentinel2_images,
    dates,
    output_path,
    band_combo="NCC",
    fps=0.5,
    stretch_percentiles=(1, 99),
    blur_factor=None
):
    """
    Create a video from Sentinel-2 image stack.
    """

    # ---------------------------
    # Band selection
    # ---------------------------
    if isinstance(band_combo, str):
        band_combo = band_combo.upper()
        if band_combo == "NCC":
            bands = [2, 1, 0]
        elif band_combo == "FCC":
            bands = [6, 2, 1]
        elif band_combo == "SWIR":
            bands = [9, 8, 7]
        else:
            raise ValueError("band_combo must be NCC, FCC, SWIR, or a custom list")
    else:
        bands = band_combo  # custom band indices
        band_combo = "CUSTOM"

    # ---------------------------
    # Blur factor defaulting logic
    # ---------------------------
    if blur_factor is None:
        if band_combo == "SWIR":
            blur_factor = 5
        else:
            blur_factor = 1

    # ---------------------------
    # Build frames
    # ---------------------------
    frames = []
    for img in sentinel2_images:
        frame = img[:, :, bands]

        frame = np.nan_to_num(
            np.clip(
                frame / np.nanpercentile(frame, stretch_percentiles[1]),
                0, 1
            ),
            0
        )

        # Optional smoothing + downsampling
        if blur_factor > 1:
            frame = cv2.blur(frame, (blur_factor, blur_factor))
            frame = frame[::blur_factor, ::blur_factor, :]

        frames.append(frame)

    frames = np.array(frames)

    # ---------------------------
    # Percentile stretching (global)
    # ---------------------------
    def stretch(stack, low, high):
        mn = np.nanpercentile(stack, low)
        mx = np.nanpercentile(stack, high)
        return (np.clip(stack, mn, mx) - mn) / (mx - mn)

    for i in range(3):
        frames[:, :, :, i] = stretch(
            frames[:, :, :, i],
            stretch_percentiles[0],
            stretch_percentiles[1]
        )

    # ---------------------------
    # Text labels
    # ---------------------------
    texts = [interval[0].split("T")[0] for interval in dates]

    # ---------------------------
    # Write video
    # ---------------------------
    create_video_from_frames(
        frames=[frames[i] for i in range(frames.shape[0])],
        texts=texts,
        output_filename=output_path,
        fps=fps
    )

