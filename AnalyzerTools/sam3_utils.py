"""
SAM3 (Segment Anything Model 3) utility functions.
"""

import os
import sam3
import torch
from sam3.model_builder import build_sam3_video_predictor


def initialize_predictor(gpus_to_use=None, use_all_gpus=True):
    """
    Initialize SAM3 video predictor.
    
    Parameters
    ----------
    gpus_to_use : list or range, optional
        Specific GPUs to use. If None and use_all_gpus=True, uses all available GPUs.
        If None and use_all_gpus=False, uses only the current GPU.
    use_all_gpus : bool
        If True and gpus_to_use is None, uses all available GPUs.
        If False and gpus_to_use is None, uses only the current GPU.
        Default: True
    
    Returns
    -------
    predictor
        Initialized SAM3 video predictor
    
    Examples
    --------
    >>> # Use all available GPUs
    >>> predictor = initialize_predictor()
    
    >>> # Use only current GPU
    >>> predictor = initialize_predictor(use_all_gpus=False)
    
    >>> # Use specific GPUs
    >>> predictor = initialize_predictor(gpus_to_use=[0, 1])
    """
    if gpus_to_use is None:
        if use_all_gpus:
            # Use all available GPUs
            gpus_to_use = range(torch.cuda.device_count())
        else:
            # Use only the current GPU
            gpus_to_use = [torch.cuda.current_device()]
    
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    
    return predictor


def get_sam3_root(sam3_path=None):
    """
    Get the root directory of the SAM3 package or cloned repository.
    
    Parameters
    ----------
    sam3_path : str, optional
        Path to cloned SAM3 directory. If None, tries to get it from
        the installed sam3 package location. Default: None
    
    Returns
    -------
    str
        Path to SAM3 root directory
    
    Examples
    --------
    >>> # Use installed package location
    >>> sam3_root = get_sam3_root()
    
    >>> # Use cloned directory
    >>> sam3_root = get_sam3_root(sam3_path="./sam3")
    """
    if sam3_path is not None:
        # Use provided path
        sam3_path = os.path.abspath(sam3_path)
        if not os.path.exists(sam3_path):
            raise ValueError(f"SAM3 path does not exist: {sam3_path}")
        return sam3_path
    else:
        # Try to get from installed package
        try:
            return os.path.abspath(os.path.join(os.path.dirname(sam3.__file__), ".."))
        except AttributeError:
            raise ValueError(
                "Could not determine SAM3 root directory. "
                "Please provide sam3_path parameter pointing to the cloned SAM3 directory."
            )

