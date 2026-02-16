#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Persistent Homology Computation Tool using CubicalRipser

This module provides functionality for computing persistent homology of images
and volumes using the CubicalRipser library. It supports various preprocessing
transformations and can handle multiple file formats.
"""

#%%
import struct
import numpy as np
import argparse
import os, time
import matplotlib.pyplot as plt
import cripser, tcripser
from PIL import Image
import re, glob
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from skimage.transform import rescale

# Optional dependency for progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Fallback function if tqdm is not available
    tqdm = lambda x: x

# Utility function to extract numeric values from filenames for sorting
num = lambda val: int(re.sub("\\D", "", val + "0"))

def comp_PH(fn):
    """
    Compute persistent homology for a single image file.

    This function is designed for batch processing of image files.
    It loads an image, converts it to grayscale, computes persistent homology
    using CubicalRipser, and saves the results as a numpy array.

    Parameters:
    -----------
    fn : str
        Filepath to the input image file

    Returns:
    --------
    None
        Results are saved to a .npy file with the same basename as input
    """
    # Load image and convert to grayscale float64 array
    im = np.array(Image.open(fn).convert('L'), dtype=np.float64)

    # Optional: Apply distance transform (currently commented out)
    # im = dt(im)

    # Compute persistent homology using default parameters
    pd = cripser.computePH(im)

    # Alternative: compute with custom parameters (currently commented out)
    # pd = cripser.computePH(im, maxdim=args.maxdim, top_dim=args.top_dim, embedded=args.embedded)

    # Save persistence diagram to numpy file
    np.save(os.path.splitext(fn)[0], pd)
    return

# load a series of files to form a volume
def load_vol(fns, transform=None, shift_value=None, threshold=None, threshold_upper_limit=None,
             scaling_factor=1, negative=False, sort=False, origin=(0,0,0)):
    """
    Load and preprocess a series of image files to create a volume for persistent homology computation.

    This function supports multiple file formats and various preprocessing transformations
    including distance transforms, thresholding, and geometric transformations.

    Parameters:
    -----------
    fns : list of str
        List of file paths to load and stack into a volume
    transform : str, optional
        Type of transformation to apply:
        - 'distance': Euclidean distance transform from background
        - 'signed_distance': Signed distance transform
        - 'distance_inv'/'signed_distance_inv': Inverted distance transforms
        - 'upward'/'downward': Height-based transforms
        - 'radial'/'radial_inv': Radial distance from origin
        - 'geodesic'/'geodesic_inv': Geodesic distance transforms
    shift_value : float, optional
        Value to add to all pixel values after processing
    threshold : float, optional
        Lower threshold for binarization before distance transform
    threshold_upper_limit : float, optional
        Upper threshold for binarization
    scaling_factor : float, default=1
        Scaling factor to resize the volume (for computational efficiency)
    negative : bool, default=False
        Whether to negate all pixel values
    sort : bool, default=False
        Whether to sort filenames numerically before loading
    origin : tuple, default=(0,0,0)
        Origin point for radial transformations (z,y,x coordinates)

    Returns:
    --------
    tuple
        (img_arr, dtype) where:
        - img_arr: numpy array of the processed volume
        - dtype: original data type of the input images
    """
    print("loading {}".format(fns[0]))

    # Sort filenames numerically if requested
    if sort:
        fns.sort(key=num)

    images = []

    # Process each file in the list
    for ffn in fns:
        fn, ext = os.path.splitext(ffn)
        ext = ext.lower()
        dtype = int  # Default data type

        # Load different file formats
        if ext == ".npy":
            # NumPy array file
            im = np.load(ffn)
            dtype = im.dtype
        elif ext == ".dcm":
            # DICOM medical image format
            ref_dicom_in = dicom.read_file(ffn, force=True)
            # ref_dicom_in.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
            im = ref_dicom_in.pixel_array + ref_dicom_in.RescaleIntercept
        elif ext == ".csv":
            # Comma-separated values file
            im = np.loadtxt(ffn, delimiter=",")
            dtype = im.dtype
        elif ext == ".nrrd":
            # Nearly Raw Raster Data format
            im, header = nrrd.read(ffn, index_order='C')
            dtype = im.dtype
        elif ext == ".complex":
            # DIPHA complex file format
            dat = open(ffn, 'rb').read()
            magic, tp, sz, dim = struct.unpack_from("qqqq", dat, 0)
            sp = struct.unpack_from("q"*dim, dat, 8*4)  # offset = 8byte x 4
            im = np.array(struct.unpack_from("d"*sz, dat, 8*(4+dim))).reshape(sp)
        elif ext == ".txt":
            from dipha2npy import convert_perseus_to_numpy
            im = convert_perseus_to_numpy(ffn)
        else:
            # Default: treat as standard image file (jpg, png, etc.)
            im = np.array(Image.open(ffn).convert('L'))

        images.append(im)

    # Stack images to form a 3D volume (z,y,x)
    img_arr = np.squeeze(np.stack(images, axis=0))

    # ====================
    # PREPROCESSING PHASE
    # ====================
    if transform is not None:
        # Step 1: Binarization (convert to binary image based on threshold)
        if threshold is not None:
            if threshold_upper_limit is not None:
                # Double threshold: keep pixels within range [threshold, threshold_upper_limit]
                img_arr = np.logical_and(img_arr >= threshold, img_arr <= threshold_upper_limit)
            else:
                # Single lower threshold
                img_arr = (img_arr >= threshold)
        elif threshold_upper_limit is not None:
            # Single upper threshold
            img_arr = (img_arr <= threshold_upper_limit)
        else:
            # Automatic threshold using Otsu's method
            img_arr = (img_arr >= threshold_otsu(img_arr))

        # Step 2: Apply geometric transformations
        if 'distance' in transform:
            # Distance transform: compute Euclidean distance from background
            if '_inv' in transform:
                # Invert the binary image before distance transform
                img_arr = ~img_arr

            # Compute distance transform from foreground
            im = distance_transform_edt(img_arr)

            if 'signed' in transform:
                # Signed distance: positive inside, negative outside
                im -= distance_transform_edt(~img_arr)

            img_arr = im

        elif transform == 'signed_distance':
            # Direct signed distance transform
            img_arr = distance_transform_edt(img_arr) - distance_transform_edt(~img_arr)

        elif transform in ['downward', 'upward']:
            # Height-based transforms using z-coordinate
            null_idx = img_arr == 0  # Remember background pixels

            # Create height coordinate array
            if len(img_arr.shape) == 3:  # 3D volume (z,y,x)
                h = np.arange(img_arr.shape[0]).reshape(-1, 1, 1)
            else:  # 2D image stack (z,y)
                h = np.arange(img_arr.shape[0]).reshape(-1, 1)

            if transform == 'upward':
                # Invert height direction (higher z = lower value)
                h = -h

            # Multiply binary mask by height values
            img_arr = (img_arr * h)
            # Set background pixels to maximum value
            img_arr[null_idx] = np.max(img_arr)

        elif 'radial' in transform:
            # Radial distance transform from specified origin
            null_idx = img_arr == 0  # Remember background pixels

            # Compute radial distance from origin to each voxel
            h = np.linalg.norm(
                np.stack(np.meshgrid(*map(range, img_arr.shape), indexing='ij'), axis=-1) - np.array(origin),
                axis=-1
            )

            # Apply radial weighting to binary mask
            img_arr = (img_arr * h)

            if transform == 'radial_inv':
                # Invert radial distance (closer to origin = higher value)
                img_arr *= -1
            else:
                # Set background to maximum radial distance
                img_arr[null_idx] = np.max(h)

        elif 'geodesic' in transform:
            # Geodesic distance transform using fast marching method
            try:
                import skfmm
            except:
                print("install skfmm by 'pip install scikit-fmm'")
                exit()

            # Initialize region of interest (ROI)
            roi = np.ones(img_arr.shape)

            # Set origin based on mask file or coordinates
            if args.origin_mask is not None:
                # Load origin mask from NRRD file
                skl, header = nrrd.read(args.origin_mask, index_order='C')
                roi[skl > 0] = 0  # >0 specifies outside the object
            else:
                # Use specified origin coordinates
                if len(roi.shape) == 3:
                    roi[args.origin[0], args.origin[1], args.origin[2]] = 0
                else:
                    roi[args.origin[0], args.origin[1]] = 0

            # Compute signed distance from origin using fast marching
            # Outside the region (~img_arr) is masked
            img_arr = skfmm.distance(np.ma.MaskedArray(roi, ~img_arr))

            if '_inv' in transform:
                # Invert geodesic distance
                img_arr *= -1

            # Fill masked values with maximum distance
            img_arr = img_arr.filled(fill_value=img_arr.max())

    # Final processing steps
    # Convert to float64 for numerical precision
    img_arr = img_arr.astype(np.float64)

    # Apply scaling if requested (for computational efficiency)
    if scaling_factor != 1:
        img_arr = rescale(img_arr, scaling_factor, order=1, mode="reflect", preserve_range=True)

    # Negate values if requested
    if negative:
        img_arr *= -1

    # Shift all values by constant if requested
    if shift_value:
        img_arr += shift_value

    print("input shape: ", img_arr.shape)
    return (img_arr, dtype)

#%%
if __name__ == "__main__":
    """
    Main execution block for persistent homology computation.

    This script can be run from command line with various options for:
    - Input data (images, volumes, numpy arrays)
    - Preprocessing transformations
    - Computation parameters
    - Output format
    """

    # ====================
    # COMMAND LINE ARGUMENT PARSING
    # ====================
    parser = argparse.ArgumentParser(
        description="Compute persistent homology using CubicalRipser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image
    python cr.py image.png

    # Directory of images with distance transform
    python cr.py images/ --transform distance --output result.npy

    # Batch processing multiple directories
    python cr.py dir1/ dir2/ --batch 4 --imgtype jpg

    # Volume with geodesic transform and custom origin
    python cr.py volume/ --transform geodesic --origin 10 20 30
        """
    )

    # Input/Output arguments
    parser.add_argument('input', type=str, nargs="*",
                       help="Input files: numpy array, multiple images, or directory containing images")
    parser.add_argument('--output', '-o', default=None,
                       help="Output file path (.npy or .csv)")
    parser.add_argument('--print', '-p', action='store_true',
                       help="Print output to console")

    # Computation parameters
    parser.add_argument('--filtration', '-f', choices=['V', 'T'], default='V',
                       help="Filtration type: V (value-based) or T (topology-based)")
    parser.add_argument('--top_dim', action='store_true',
                       help="Compute top-dimensional homology")
    parser.add_argument('--embedded', '-e', action='store_true',
                       help='Compute for the Alexander dual complex')
    parser.add_argument('--maxdim', '-m', default=3, type=int,
                       help="Maximum dimension for homology computation (default: 2)")
    parser.add_argument('--software', type=str, default="cubicalripser",
                       choices=["cubicalripser", "gudhi"],
                       help="Software backend for computation")

    # Data preprocessing arguments
    parser.add_argument('--negative', '-n', action='store_true',
                       help='Negate all pixel values')
    parser.add_argument('--sort', '-s', action='store_true',
                       help="Sort file names numerically before stacking")
    parser.add_argument('--transform', '-tr',
                       choices=[None, 'distance', 'signed_distance', 'distance_inv', 'signed_distance_inv',
                               'radial', 'radial_inv', 'geodesic', 'geodesic_inv', 'upward', 'downward'],
                       help="Preprocessing transformation to apply")
    parser.add_argument('--shift_value', '-sv', default=None, type=float,
                       help='Constant value added to all pixels after processing')
    parser.add_argument('--scaling_factor', '-sf', default=1, type=float,
                       help="Scaling factor for computational efficiency (default: 1)")

    # Thresholding arguments
    parser.add_argument('--threshold', '-th', type=float, default=None,
                       help='Lower threshold for binarization (used with transforms)')
    parser.add_argument('--threshold_upper_limit', '-thu', type=float, default=None,
                       help='Upper threshold for binarization')

    # Geometric transform arguments
    parser.add_argument('--origin', type=int, nargs='*', default=(0, 0),
                       help='Origin coordinates for radial transformation (z,y,x order)')
    parser.add_argument('--origin_mask', '-om', type=str, default=None,
                       help='NRRD file specifying origin region for geodesic transforms')

    # Batch processing arguments
    parser.add_argument('--batch', '-b', type=int, default=0,
                       help='Enable batch processing with specified number of threads (0=disabled)')
    parser.add_argument('--imgtype', '-it', type=str, default=None,
                       help='Image file extension for batch processing (default: png)')

    args = parser.parse_args()

    # ====================
    # BATCH PROCESSING MODE
    # ====================
    if args.batch > 0:
        """
        Batch processing mode: process multiple image files in parallel
        Each image is processed independently and results saved separately
        """
        fns = []

        # Default image type for batch processing
        if args.imgtype is None:
            args.imgtype = "png"

        # Collect all image files from input directories
        for dn in args.input:
            fns.extend(glob.glob(os.path.join(dn, "**/*.{}".format(args.imgtype)), recursive=True))

        # Process images in parallel using multiprocessing
        pool = Pool(args.batch)
        # Alternative: ProcessPoolExecutor approach (commented out)
        # with ProcessPoolExecutor(args.batch) as executor:
        #     tqdm(executor.map(comp_PH, fns), total=len(fns))

        print(f"Processing {len(fns)} images using {args.batch} threads...")
        with tqdm(total=len(fns)) as t:
            for _ in pool.imap_unordered(comp_PH, fns):
                t.update(1)

        print("Batch processing completed!")
        exit(0)

    # ====================
    # SINGLE VOLUME PROCESSING MODE
    # ====================

    # Handle directory input
    if os.path.isdir(args.input[0]):
        # Set default output name based on directory name
        if args.output is None:
            args.output = os.path.basename(os.path.normpath(args.input[0]))

        # Collect files from directory
        if args.imgtype is not None:
            # Search for specific image type recursively
            fns = glob.glob(os.path.join(args.input[0], "**/*.{}".format(args.imgtype)), recursive=True)
        else:
            # Get all files in directory
            fns = os.listdir(args.input[0])
            fns = [os.path.join(args.input[0], f) for f in fns]
    else:
        # Direct file list input
        fns = args.input

    # Check file format and import required dependencies
    fn, ext = os.path.splitext(fns[0])
    ext = ext.lower()

    if ext == ".dcm":
        # DICOM format requires pydicom
        try:
            import pydicom as dicom
        except ImportError:
            print("Install pydicom first by: pip install pydicom")
            exit()

    if ext == ".nrrd":
        # NRRD format requires pynrrd
        try:
            import nrrd
        except ImportError:
            print("Install nrrd first by: pip install pynrrd")
            exit()

    # Load and preprocess the volume
    img_arr, dtype = load_vol(
        fns, args.transform, args.shift_value, args.threshold,
        args.threshold_upper_limit, args.scaling_factor,
        args.negative, args.sort, origin=args.origin
    )

    # ====================
    # PERSISTENT HOMOLOGY COMPUTATION
    # ====================
    print("Computing persistent homology for {}...".format(fns[0]))
    start = time.time()

    if args.software == "gudhi":
        # Use GUDHI library for computation
        try:
            import gudhi
        except ImportError:
            print("Install gudhi first by: conda install -c conda-forge gudhi")
            exit()

        print("Computing PH with GUDHI")
        # Create cubical complex from the image array
        if args.filtration == 'V':
            gd = gudhi.CubicalComplex(vertices=img_arr)
        else:
            gd = gudhi.CubicalComplex(top_dimensional_cells=img_arr)

        # Compute persistence with coefficient field Z/2Z
        pers = gd.persistence(2, 0)
        res = np.array([[d, b, de] for d, (b, de) in pers], dtype=float)

        # Display Betti numbers (connectivity information)
        print("Betti numbers: ", gd.persistent_betti_numbers(np.inf, -np.inf))

    else:  # Default: use CubicalRipser
        # Use CubicalRipser library for computation
        if args.filtration == 'V':
            # V-construction (value-based filtration)
            res = cripser.computePH(
                img_arr,
                maxdim=args.maxdim,
                top_dim=args.top_dim,
                embedded=args.embedded
            )
        else:
            # T-construction (topology-based filtration)
            res = tcripser.computePH(
                img_arr,
                maxdim=args.maxdim,
                top_dim=args.top_dim,
                embedded=args.embedded
            )
        # Display Betti numbers for dimensions 0, 1, 2
        print("Betti numbers: ", [res[res[:, 0] == i].shape[0] for i in range(args.maxdim + 1)])

    print("Computation took: {:.2f} seconds".format(time.time() - start))

    # ====================
    # SAVE RESULTS
    # ====================
    if args.output is not None:
        if args.output.endswith(".csv"):
            # Save as CSV with specific formatting
            fmt_add = "" if args.software == "gudhi" else "%d,%d,%d,%d,%d,%d"
            if img_arr.ndim < 4:
                header = 'dim,birth,death,birth_x,birth_y,birth_z,death_x,death_y,death_z'
            else:
                header = 'dim,birth,death,birth_x,birth_y,birth_z,birth_w,death_x,death_y,death_z,death_w'
                if args.software != "gudhi":
                    fmt_add += ',%d,%d'
            np.savetxt(
                args.output, res,
                delimiter=',',
                fmt='%i,%18.10f,%18.10f,'+fmt_add,
                header=header
            )
        else:
            # Save as NumPy binary format (.npy)
            np.save(args.output, res)

        print(f"Results saved to: {args.output}")

    if args.print:
        print(res)

    print("Persistent homology computation completed successfully!")
