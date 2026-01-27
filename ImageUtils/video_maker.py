import os
import numpy as np
import tempfile
import cv2
import matplotlib.pyplot as plt
import zipfile
from datetime import datetime
from PIL import Image

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


def create_all_videos_and_zip(
    sentinel2_images,
    dates,
    output_dir="./",
    base_name=None,
    fps=0.5,
    stretch_percentiles=(1, 99),
    blur_factor=None
):
    """
    Create NCC, FCC, and SWIR videos from Sentinel-2 images and zip them together.
    
    Parameters
    ----------
    sentinel2_images : list of np.ndarray
        List of Sentinel-2 images, each with shape (H, W, 10)
    dates : list of tuple
        List of date tuples for each image
    output_dir : str
        Directory to save videos and zip file (default: current directory)
    base_name : str, optional
        Base name for output files. If None, uses timestamp
    fps : float
        Frames per second for videos (default: 0.5)
    stretch_percentiles : tuple
        Percentiles for stretching (default: (1, 99))
    blur_factor : int, optional
        Blur factor for videos. If None, uses defaults (1 for NCC/FCC, 5 for SWIR)
    
    Returns
    -------
    video_paths : dict
        Dictionary with keys 'ncc', 'fcc', 'swir', 'zip' containing file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base name if not provided
    if base_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"s2_videos_{timestamp}"
    
    # Define video file paths
    video_paths = {
        'ncc': os.path.join(output_dir, f"{base_name}_NCC.mp4"),
        'fcc': os.path.join(output_dir, f"{base_name}_FCC.mp4"),
        'swir': os.path.join(output_dir, f"{base_name}_SWIR.mp4")
    }
    
    # Create NCC video
    print("Creating NCC (Natural Color Composite) video...")
    video_maker(
        sentinel2_images=sentinel2_images,
        dates=dates,
        output_path=video_paths['ncc'],
        band_combo="NCC",
        fps=fps,
        stretch_percentiles=stretch_percentiles,
        blur_factor=blur_factor if blur_factor is not None else 1
    )
    print(f"  ✓ NCC video saved: {video_paths['ncc']}")
    
    # Create FCC video
    print("\nCreating FCC (False Color Composite) video...")
    video_maker(
        sentinel2_images=sentinel2_images,
        dates=dates,
        output_path=video_paths['fcc'],
        band_combo="FCC",
        fps=fps,
        stretch_percentiles=stretch_percentiles,
        blur_factor=blur_factor if blur_factor is not None else 1
    )
    print(f"  ✓ FCC video saved: {video_paths['fcc']}")
    
    # Create SWIR video
    print("\nCreating SWIR (Shortwave Infrared) video...")
    video_maker(
        sentinel2_images=sentinel2_images,
        dates=dates,
        output_path=video_paths['swir'],
        band_combo="SWIR",
        fps=fps,
        stretch_percentiles=stretch_percentiles,
        blur_factor=blur_factor if blur_factor is not None else 1
    )
    print(f"  ✓ SWIR video saved: {video_paths['swir']}")
    
    # Create zip file
    zip_path = os.path.join(output_dir, f"{base_name}_all_videos.zip")
    print(f"\nCreating zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all three videos to the zip
        for video_type, video_path in video_paths.items():
            if os.path.exists(video_path):
                # Add file with just the filename (not full path) to zip
                zipf.write(video_path, os.path.basename(video_path))
                print(f"  ✓ Added {video_type.upper()} video to zip")
            else:
                print(f"  ✗ Warning: {video_path} not found, skipping")
    
    video_paths['zip'] = zip_path
    print(f"\n✓ All videos zipped successfully: {zip_path}")
    print(f"  Original videos are preserved in: {output_dir}")
    
    return video_paths


def image_maker(
    sentinel2_images,
    dates,
    output_folder,
    band_combo="NCC",
    stretch_percentiles=(1, 99),
    blur_factor=None,
    filename_prefix="",
    add_date_to_filename=True,
    sam3_format=False
):
    """
    Create a folder with JPG images (3 bands) from a stack of Sentinel-2 images.
    
    Parameters
    ----------
    sentinel2_images : list of np.ndarray
        List of Sentinel-2 images, each with shape (H, W, 10)
    dates : list of tuple
        List of date tuples for each image
    output_folder : str
        Folder path to save JPG images
    band_combo : str or list
        Band combination: "NCC", "FCC", "SWIR", or custom list of 3 band indices
    stretch_percentiles : tuple
        Percentiles for stretching (default: (1, 99))
    blur_factor : int, optional
        Blur factor for images. If None, uses default: 1 for all band combinations
    filename_prefix : str
        Prefix for output filenames (default: ""). Ignored if sam3_format=True
    add_date_to_filename : bool
        Whether to add date to filename (default: True). Ignored if sam3_format=True
    sam3_format : bool
        If True, saves files as "<frame_index>.jpg" format (e.g., "0.jpg", "1.jpg")
        for SAM3 compatibility. If False, uses prefix/date format (default: False)
    
    Returns
    -------
    saved_files : list
        List of paths to saved JPG files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
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
        if len(bands) != 3:
            raise ValueError("band_combo must contain exactly 3 band indices")
    
    # ---------------------------
    # Blur factor defaulting logic
    # ---------------------------
    if blur_factor is None:
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
    # Text labels and filenames
    # ---------------------------
    texts = [interval[0].split("T")[0] for interval in dates]
    
    # ---------------------------
    # Save images as JPG
    # ---------------------------
    saved_files = []
    
    for i, frame in enumerate(frames):
        # Convert to uint8 (0-255 range)
        frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        
        # Create filename
        if sam3_format:
            # SAM3 format: simple "<frame_index>.jpg" (e.g., "0.jpg", "1.jpg")
            filename = f"{i}.jpg"
        elif add_date_to_filename:
            date_str = texts[i].replace("-", "")
            if filename_prefix:
                filename = f"{filename_prefix}_{date_str}_{i:04d}.jpg"
            else:
                filename = f"{date_str}_{i:04d}.jpg"
        else:
            if filename_prefix:
                filename = f"{filename_prefix}_{i:04d}.jpg"
            else:
                filename = f"image_{i:04d}.jpg"
        
        filepath = os.path.join(output_folder, filename)
        
        # Frames are already in RGB order (from band selection)
        # PIL's Image.fromarray expects RGB order, so no conversion needed
        # Save as JPG using PIL
        img_pil = Image.fromarray(frame_uint8)
        img_pil.save(filepath, "JPEG", quality=95)
        
        saved_files.append(filepath)
    
    print(f"Saved {len(saved_files)} images to: {os.path.abspath(output_folder)}")
    
    return saved_files

