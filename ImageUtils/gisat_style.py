"""
GISAT-style image processing utilities.
"""

import numpy as np
import cv2


def gisat_style(sentinel2_images, preserve_bands=[0, 1, 2, 6], degradation_factor=5):
    """
    Apply GISAT-style processing to Sentinel-2 images.
    
    Preserves specified bands (RGB and NIR) at full resolution while degrading
    other bands by downsampling and upsampling them.
    
    Parameters
    ----------
    sentinel2_images : list of np.ndarray or np.ndarray
        List of Sentinel-2 images, each with shape (H, W, 10), or
        a single numpy array with shape (N, H, W, 10) or (H, W, 10)
    preserve_bands : list of int
        Band indices to preserve at full resolution (default: [0, 1, 2, 6] for RGB and NIR)
    degradation_factor : int
        Factor by which to downsample/upsample degraded bands (default: 5)
    
    Returns
    -------
    processed_images : list of np.ndarray or np.ndarray
        Processed images with same shape as input, where preserved bands remain
        unchanged and other bands are degraded by the specified factor
    
    Examples
    --------
    >>> images = [img1, img2, img3]  # Each with shape (100, 100, 10)
    >>> processed = gisat_style(images)
    >>> # Bands 0, 1, 2, 6 are preserved, others are degraded by factor of 5
    
    >>> # Custom preserve bands
    >>> processed = gisat_style(images, preserve_bands=[0, 1, 2, 6, 8])
    """
    # Handle different input formats
    is_list = isinstance(sentinel2_images, list)
    is_single = not is_list and len(sentinel2_images.shape) == 3
    
    if is_list:
        images = sentinel2_images
    elif is_single:
        images = [sentinel2_images]
    else:
        # Assume shape (N, H, W, 10)
        images = [sentinel2_images[i] for i in range(sentinel2_images.shape[0])]
    
    if not images:
        raise ValueError("Input images list cannot be empty")
    
    # Get dimensions from first image
    first_img = images[0]
    if len(first_img.shape) != 3:
        raise ValueError(f"Expected images with shape (H, W, bands), got {first_img.shape}")
    
    H, W, num_bands = first_img.shape
    
    if num_bands != 10:
        raise ValueError(f"Expected 10 bands for Sentinel-2, got {num_bands}")
    
    # Validate preserve_bands
    preserve_bands = list(set(preserve_bands))  # Remove duplicates
    for band_idx in preserve_bands:
        if band_idx < 0 or band_idx >= num_bands:
            raise ValueError(f"Band index {band_idx} out of range (0-{num_bands-1})")
    
    # Determine which bands to degrade
    all_bands = set(range(num_bands))
    degrade_bands = sorted(list(all_bands - set(preserve_bands)))
    
    processed_images = []
    
    for img in images:
        if img.shape != (H, W, num_bands):
            raise ValueError(f"Image shape {img.shape} doesn't match expected shape ({H}, {W}, {num_bands})")
        
        # Create output array
        processed_img = np.zeros_like(img)
        
        # Process each band
        for band_idx in range(num_bands):
            band_data = img[:, :, band_idx]
            
            if band_idx in preserve_bands:
                # Preserve band at full resolution
                processed_img[:, :, band_idx] = band_data
            else:
                # Degrade band: downsample then upsample
                # Calculate new dimensions
                new_H = max(1, H // degradation_factor)
                new_W = max(1, W // degradation_factor)
                
                # Handle NaN values by replacing with 0 for interpolation
                band_data_clean = np.nan_to_num(band_data, nan=0.0)
                
                # Downsample using INTER_AREA (best for downsampling)
                downsampled = cv2.resize(
                    band_data_clean,
                    (new_W, new_H),
                    interpolation=cv2.INTER_AREA
                )
                
                # Upsample back to original size using INTER_LINEAR
                upsampled = cv2.resize(
                    downsampled,
                    (W, H),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Restore NaN values if they existed in original
                if np.any(np.isnan(band_data)):
                    upsampled = np.where(np.isnan(band_data), np.nan, upsampled)
                
                processed_img[:, :, band_idx] = upsampled
        
        processed_images.append(processed_img)
    
    # Return in same format as input
    if is_list:
        return processed_images
    elif is_single:
        return processed_images[0]
    else:
        # Return as numpy array with shape (N, H, W, 10)
        return np.array(processed_images)


