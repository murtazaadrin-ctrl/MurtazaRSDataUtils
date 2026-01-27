"""
SAM3 bidirectional video segmentation with text prompts.

This module provides functionality for bi-directional, user-seeded video segmentation
workflow where a user can select a frame and text prompt, and the system will:
1. Propagate forward from the seed frame
2. Propagate backward from the seed frame
3. Merge the results for unified per-frame masks
"""

import os
import glob
import shutil
from typing import Dict, List, Optional, Tuple, Any


def make_reversed_video_folder(src_folder: str, dst_folder: str) -> int:
    """
    Create a reversed copy of video frames for backward propagation.
    
    Parameters
    ----------
    src_folder : str
        Source folder containing frames (e.g., "0.jpg", "1.jpg", ...)
    dst_folder : str
        Destination folder for reversed frames
    
    Returns
    -------
    int
        Total number of frames
    
    Examples
    --------
    >>> num_frames = make_reversed_video_folder("./frames", "./frames_reversed")
    >>> # Creates reversed copy: frame 0.jpg becomes (N-1).jpg, etc.
    """
    os.makedirs(dst_folder, exist_ok=True)
    
    # Get all frames and sort by integer index
    frames = sorted(
        glob.glob(os.path.join(src_folder, "*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    
    if not frames:
        raise ValueError(f"No .jpg files found in {src_folder}")
    
    N = len(frames)
    
    # Copy frames in reverse order
    for i, src in enumerate(reversed(frames)):
        dst = os.path.join(dst_folder, f"{i}.jpg")
        shutil.copy(src, dst)
    
    return N


def propagate_in_video(predictor, session_id: str) -> Dict[int, Any]:
    """
    Propagate masks forward in video from current session state.
    
    Parameters
    ----------
    predictor
        SAM3 predictor instance
    session_id : str
        Session ID for the video session
    
    Returns
    -------
    Dict[int, Any]
        Dictionary mapping frame_index to outputs
    """
    outputs_per_frame = {}
    
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    
    return outputs_per_frame


def run_forward_from_seed(
    predictor,
    video_path: str,
    seed_frame: int,
    text_prompt: str
) -> Dict[int, Any]:
    """
    Run forward propagation from a seed frame with text prompt.
    
    Parameters
    ----------
    predictor
        SAM3 predictor instance
    video_path : str
        Path to video folder containing frames
    seed_frame : int
        Frame index to seed the prompt (0-based)
    text_prompt : str
        Text description of object to track (e.g., "person", "car")
    
    Returns
    -------
    Dict[int, Any]
        Dictionary mapping frame_index to outputs (forward propagation results)
    """
    # Start session
    resp = predictor.handle_request(
        dict(type="start_session", resource_path=video_path)
    )
    session_id = resp["session_id"]
    
    # Reset session
    predictor.handle_request(
        dict(type="reset_session", session_id=session_id)
    )
    
    # Add text prompt at seed frame
    predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=seed_frame,
        text=text_prompt,
    ))
    
    # Propagate forward
    forward_outputs = propagate_in_video(predictor, session_id)
    
    return forward_outputs


def run_backward_via_reverse(
    predictor,
    original_video_path: str,
    reversed_video_path: str,
    seed_frame: int,
    total_frames: int,
    text_prompt: str
) -> Dict[int, Any]:
    """
    Run backward propagation via reversed video trick.
    
    This works even if the API only supports forward propagation by:
    1. Creating a reversed copy of frames
    2. Running forward propagation on reversed frames
    3. Mapping results back to original frame indices
    
    Parameters
    ----------
    predictor
        SAM3 predictor instance
    original_video_path : str
        Path to original video folder
    reversed_video_path : str
        Path to reversed video folder (will be created if needed)
    seed_frame : int
        Original frame index where prompt was seeded
    total_frames : int
        Total number of frames in video
    text_prompt : str
        Text description of object to track
    
    Returns
    -------
    Dict[int, Any]
        Dictionary mapping original frame_index to outputs (backward propagation results)
    """
    # Calculate seed frame in reversed video
    reversed_seed = total_frames - 1 - seed_frame
    
    # Ensure reversed folder exists
    if not os.path.exists(reversed_video_path) or not glob.glob(os.path.join(reversed_video_path, "*.jpg")):
        make_reversed_video_folder(original_video_path, reversed_video_path)
    
    # Start session with reversed video
    resp = predictor.handle_request(
        dict(type="start_session", resource_path=reversed_video_path)
    )
    session_id = resp["session_id"]
    
    # Reset session
    predictor.handle_request(
        dict(type="reset_session", session_id=session_id)
    )
    
    # Add text prompt at reversed seed frame
    predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=reversed_seed,
        text=text_prompt,
    ))
    
    # Propagate forward in reversed video
    reversed_outputs = propagate_in_video(predictor, session_id)
    
    # Map back to original indices
    backward_outputs = {}
    for rev_idx, out in reversed_outputs.items():
        orig_idx = total_frames - 1 - rev_idx
        backward_outputs[orig_idx] = out
    
    return backward_outputs


def merge_bidirectional_outputs(
    forward: Dict[int, Any],
    backward: Dict[int, Any],
    merge_strategy: str = "prefer_forward"
) -> Dict[int, Any]:
    """
    Merge forward and backward propagation results.
    
    Parameters
    ----------
    forward : Dict[int, Any]
        Forward propagation outputs (frame_index -> outputs)
    backward : Dict[int, Any]
        Backward propagation outputs (frame_index -> outputs)
    merge_strategy : str
        Strategy for merging when both forward and backward exist:
        - "prefer_forward": Use forward result (default)
        - "prefer_backward": Use backward result
        - "union": Union of masks (if outputs contain masks)
        - "intersection": Intersection of masks (if outputs contain masks)
    
    Returns
    -------
    Dict[int, Any]
        Merged outputs for all frames
    """
    merged = {}
    
    all_frames = set(forward.keys()) | set(backward.keys())
    
    for frame_idx in sorted(all_frames):
        if frame_idx in forward and frame_idx in backward:
            # Both forward and backward have results for this frame
            if merge_strategy == "prefer_forward":
                merged[frame_idx] = forward[frame_idx]
            elif merge_strategy == "prefer_backward":
                merged[frame_idx] = backward[frame_idx]
            elif merge_strategy == "union":
                # Try to merge masks if they exist
                # This is a placeholder - actual implementation depends on output format
                merged[frame_idx] = forward[frame_idx]  # Default to forward
            elif merge_strategy == "intersection":
                merged[frame_idx] = forward[frame_idx]  # Default to forward
            else:
                raise ValueError(f"Unknown merge_strategy: {merge_strategy}")
        elif frame_idx in forward:
            merged[frame_idx] = forward[frame_idx]
        else:
            merged[frame_idx] = backward[frame_idx]
    
    return merged


def run_bidirectional_tracking(
    predictor,
    video_path: str,
    seed_frame: int,
    text_prompt: str,
    reversed_video_path: Optional[str] = None,
    merge_strategy: str = "prefer_forward",
    create_reversed: bool = True
) -> Dict[int, Any]:
    """
    Run bidirectional tracking from a seed frame with text prompt.
    
    This is the main function that:
    1. Runs forward propagation from seed frame
    2. Runs backward propagation via reversed video
    3. Merges the results
    
    Parameters
    ----------
    predictor
        SAM3 predictor instance
    video_path : str
        Path to video folder containing frames (e.g., "0.jpg", "1.jpg", ...)
    seed_frame : int
        Frame index to seed the prompt (0-based)
    text_prompt : str
        Text description of object to track (e.g., "person", "car", "building")
    reversed_video_path : str, optional
        Path to reversed video folder. If None, will be created automatically
        as "{video_path}_reversed"
    merge_strategy : str
        Strategy for merging forward and backward results:
        - "prefer_forward": Use forward result when both exist (default)
        - "prefer_backward": Use backward result when both exist
        - "union": Union of masks (if applicable)
        - "intersection": Intersection of masks (if applicable)
    create_reversed : bool
        Whether to create reversed video folder if it doesn't exist (default: True)
    
    Returns
    -------
    Dict[int, Any]
        Merged outputs for all frames (frame_index -> outputs)
    
    Examples
    --------
    >>> merged_outputs = run_bidirectional_tracking(
    ...     predictor=predictor,
    ...     video_path="./images/NCC",
    ...     seed_frame=10,
    ...     text_prompt="person"
    ... )
    >>> # merged_outputs contains masks for all frames
    """
    # Count total frames
    frames = sorted(
        glob.glob(os.path.join(video_path, "*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    total_frames = len(frames)
    
    if seed_frame < 0 or seed_frame >= total_frames:
        raise ValueError(f"seed_frame {seed_frame} out of range [0, {total_frames-1}]")
    
    # Set up reversed video path
    if reversed_video_path is None:
        reversed_video_path = f"{video_path}_reversed"
    
    # Create reversed folder if needed
    if create_reversed and (not os.path.exists(reversed_video_path) or 
                           not glob.glob(os.path.join(reversed_video_path, "*.jpg"))):
        make_reversed_video_folder(video_path, reversed_video_path)
    
    # Run forward propagation
    forward_outputs = run_forward_from_seed(
        predictor, video_path, seed_frame, text_prompt
    )
    
    # Run backward propagation via reverse
    backward_outputs = run_backward_via_reverse(
        predictor,
        video_path,
        reversed_video_path,
        seed_frame,
        total_frames,
        text_prompt,
    )
    
    # Merge results
    merged = merge_bidirectional_outputs(
        forward_outputs, backward_outputs, merge_strategy=merge_strategy
    )
    
    return merged


def prepare_masks_for_visualization(
    outputs: Dict[int, Any],
    format_type: str = "default"
) -> Dict[int, Any]:
    """
    Prepare mask outputs for visualization.
    
    This is a wrapper that can be customized based on your visualization needs.
    You may want to import the actual function from sam3.visualization_utils.
    
    Parameters
    ----------
    outputs : Dict[int, Any]
        Dictionary of frame_index -> outputs
    format_type : str
        Format type for visualization (default: "default")
    
    Returns
    -------
    Dict[int, Any]
        Formatted outputs ready for visualization
    """
    # This is a placeholder - you may want to use the actual function from sam3
    # from sam3.visualization_utils import prepare_masks_for_visualization as sam3_prepare
    # return {idx: sam3_prepare(out) for idx, out in outputs.items()}
    
    return outputs


def get_frame_mask(
    merged_outputs: Dict[int, Any],
    frame_index: int
) -> Optional[Any]:
    """
    Get mask for a specific frame from merged outputs.
    
    Useful for UI integration where user scrubs through frames.
    
    Parameters
    ----------
    merged_outputs : Dict[int, Any]
        Merged outputs from run_bidirectional_tracking
    frame_index : int
        Frame index to retrieve
    
    Returns
    -------
    Optional[Any]
        Mask/output for the frame, or None if not found
    
    Examples
    --------
    >>> # In UI loop
    >>> current_frame = 15
    >>> overlay = get_frame_mask(merged_outputs, current_frame)
    >>> if overlay:
    ...     render(frame, overlay)
    """
    return merged_outputs.get(frame_index, None)

