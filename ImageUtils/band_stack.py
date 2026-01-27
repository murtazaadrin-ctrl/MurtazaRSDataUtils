"""
Band stacking utilities for satellite imagery.
"""

import numpy as np


def create_band_stack(images, band_index):
    """
    Create a stack of a specific band/index from a list of multi-band images.
    
    Extracts the specified band from each image and stacks them into a 4D array
    with shape [N, 1, H, W] where N is the number of images.
    
    Parameters
    ----------
    images : list of np.ndarray
        List of images, each with shape (H, W, bands) or (H, W)
        For Sentinel-2: each image should have shape (H, W, 10)
    band_index : int
        Index of the band to extract (0-based)
        For Sentinel-2:
            - 0: B02 (Blue)
            - 1: B03 (Green)
            - 2: B04 (Red)
            - 3: B05 (Red Edge 1)
            - 4: B06 (Red Edge 2)
            - 5: B07 (Red Edge 3)
            - 6: B08 (NIR)
            - 7: B8A (NIR Narrow)
            - 8: B11 (SWIR 1)
            - 9: B12 (SWIR 2)
    
    Returns
    -------
    stack : np.ndarray
        Stacked band data with shape [N, 1, H, W]
        where N is the number of images
    
    Examples
    --------
    >>> images = [img1, img2, img3]  # Each with shape (100, 100, 10)
    >>> red_stack = create_band_stack(images, band_index=2)  # Extract Red band
    >>> print(red_stack.shape)  # (3, 1, 100, 100)
    """
    if not images:
        raise ValueError("Images list cannot be empty")
    
    # Extract the specified band from the first image to get dimensions
    first_image = images[0]
    
    if len(first_image.shape) == 2:
        # Single band image
        H, W = first_image.shape
        if band_index != 0:
            raise ValueError(f"Band index {band_index} out of range for single-band image")
        band_data = first_image
    elif len(first_image.shape) == 3:
        # Multi-band image (H, W, bands)
        H, W, num_bands = first_image.shape
        if band_index < 0 or band_index >= num_bands:
            raise ValueError(f"Band index {band_index} out of range. Image has {num_bands} bands (0-{num_bands-1})")
        band_data = first_image[:, :, band_index]
    else:
        raise ValueError(f"Unsupported image shape: {first_image.shape}. Expected (H, W) or (H, W, bands)")
    
    # Initialize the stack with shape [N, 1, H, W]
    N = len(images)
    stack = np.zeros((N, 1, H, W), dtype=band_data.dtype)
    
    # Fill the stack
    for i, image in enumerate(images):
        if image.shape[:2] != (H, W):
            raise ValueError(f"Image {i} has shape {image.shape}, expected height={H}, width={W}")
        
        if len(image.shape) == 2:
            # Single band image
            if band_index != 0:
                raise ValueError(f"Band index {band_index} out of range for single-band image {i}")
            stack[i, 0, :, :] = image
        elif len(image.shape) == 3:
            # Multi-band image
            if band_index < 0 or band_index >= image.shape[2]:
                raise ValueError(f"Band index {band_index} out of range for image {i} with {image.shape[2]} bands")
            stack[i, 0, :, :] = image[:, :, band_index]
        else:
            raise ValueError(f"Unsupported image shape for image {i}: {image.shape}")
    
    return stack


def create_multi_band_stack(images, band_indices):
    """
    Create a stack of multiple bands from a list of multi-band images.
    
    Extracts the specified bands from each image and stacks them into a 4D array
    with shape [N, C, H, W] where N is the number of images and C is the number of bands.
    
    Parameters
    ----------
    images : list of np.ndarray
        List of images, each with shape (H, W, bands)
    band_indices : list of int
        List of band indices to extract (0-based)
    
    Returns
    -------
    stack : np.ndarray
        Stacked band data with shape [N, C, H, W]
        where N is the number of images and C is the number of selected bands
    
    Examples
    --------
    >>> images = [img1, img2, img3]  # Each with shape (100, 100, 10)
    >>> rgb_stack = create_multi_band_stack(images, band_indices=[2, 1, 0])  # RGB
    >>> print(rgb_stack.shape)  # (3, 3, 100, 100)
    """
    if not images:
        raise ValueError("Images list cannot be empty")
    
    if not band_indices:
        raise ValueError("Band indices list cannot be empty")
    
    # Extract dimensions from the first image
    first_image = images[0]
    if len(first_image.shape) != 3:
        raise ValueError(f"Expected multi-band images with shape (H, W, bands), got {first_image.shape}")
    
    H, W, num_bands = first_image.shape
    
    # Validate band indices
    for band_idx in band_indices:
        if band_idx < 0 or band_idx >= num_bands:
            raise ValueError(f"Band index {band_idx} out of range. Image has {num_bands} bands (0-{num_bands-1})")
    
    # Initialize the stack with shape [N, C, H, W]
    N = len(images)
    C = len(band_indices)
    stack = np.zeros((N, C, H, W), dtype=first_image.dtype)
    
    # Fill the stack
    for i, image in enumerate(images):
        if image.shape[:2] != (H, W):
            raise ValueError(f"Image {i} has shape {image.shape}, expected height={H}, width={W}")
        if image.shape[2] != num_bands:
            raise ValueError(f"Image {i} has {image.shape[2]} bands, expected {num_bands}")
        
        for c, band_idx in enumerate(band_indices):
            stack[i, c, :, :] = image[:, :, band_idx]
    
    return stack


