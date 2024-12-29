import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.stats import linregress, norm
# Load the FITS file
filepath = "/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits"  # Replace with your FITS file path
with fits.open(filepath) as hdul:
    image_data = hdul[0].data
    header = hdul[0].header

# Read calibration values from the FITS header
MAGZPT = header.get('MAGZPT', 25.0)  # Instrumental zero point (default 25.0)
MAGZRR = header.get('MAGZRR', 0.02)  # Error on zero point (default 0.02)

# Step 1: Mask edges and bright pixels
masked_image = np.copy(image_data).astype(float)  # Ensure it's float for masking
edge_margin = 85  # Smaller margin for edge masking
masked_image[:edge_margin, :] = np.nan  # Mask top edge
masked_image[-edge_margin:, :] = np.nan  # Mask bottom edge
masked_image[:, :edge_margin] = np.nan  # Mask left edge
masked_image[:, -edge_margin:] = np.nan  # Mask right edge
masked_image[masked_image > 50000] = np.nan  # Mask very bright pixels (less aggressive)

# Step 2: Replace NaNs with median for processing
median = np.nanmedian(masked_image)  # Median excluding NaN values
clean_image = np.nan_to_num(masked_image, nan=median)

# Step 3: Compute background statistics
mean, median, std = sigma_clipped_stats(clean_image, sigma=3.0)
print(f"Background Mean: {mean}, Median: {median}, Std Dev: {std}")

# Step 4: Detect sources using segmentation
from photutils.segmentation import detect_sources

threshold = median + (5 * std)  # Lower detection threshold for fainter sources
segmentation_map = detect_sources(clean_image, threshold, npixels=5)  # Minimum size = 5 pixels

if segmentation_map is None:
    print("No sources were detected.")
    exit()

print(f"Number of sources detected: {np.max(segmentation_map.data)}")

# Step 5: Perform photometry
catalog = SourceCatalog(clean_image, segmentation_map)
positions = np.transpose((catalog.xcentroid, catalog.ycentroid))
apertures = CircularAperture(positions, r=6)  # Aperture radius = 5 pixels
phot_table = aperture_photometry(clean_image, apertures)

# Convert counts to magnitudes
phot_table['mag'] = MAGZPT - 2.5 * np.log10(phot_table['aperture_sum'])

# Step 6: Compute cumulative counts for logN(m)
mags = phot_table['mag']
mag_bins = np.arange(np.min(mags), np.max(mags), 0.1)  # Magnitude bins
N_m = np.array([np.sum(mags <= m) for m in mag_bins])
valid_bins = N_m > 0  # Only keep bins with sources
logN_m = np.log10(N_m[valid_bins])
mag_bins = mag_bins[valid_bins]

# Handle zero counts
valid_bins = np.array(N_m) > 0
logN_m = np.log10(np.array(N_m)[valid_bins])
mag_bins = mag_bins[valid_bins]

# Handle zero counts
valid_bins = N_m > 0  # Only keep bins with sources
mag_bins = mag_bins[valid_bins]
N_m = N_m[valid_bins]

# Compute logN(m) and errors
logN_m = np.log10(N_m)
logN_m_error = 1 / (np.sqrt(N_m) * np.log(10))  # Error in log10 scale

# Fit a line to logN(m)
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(mag_bins, logN_m)

# Plot logN(m) vs m with error bars and fitted line
plt.figure(figsize=(8, 6))
plt.errorbar(mag_bins, logN_m, yerr=logN_m_error, fmt='o', color='blue', ecolor='gray', label='Measured Data with Errors', alpha=0.7)
plt.plot(mag_bins, slope * mag_bins + intercept, '--', label=f'Fitted Line: logN(m) = {slope:.2f}m + {intercept:.2f}', color='red')
plt.xlabel('Magnitude (m)')
plt.ylabel('log N(<m)')
plt.title('Galaxy Number Counts with Error Bars')
plt.legend()
plt.grid()
plt.show()

# Display fitted line parameters
print(f"Fitted line parameters:")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")

# Fit a line to logN(m)
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(mag_bins, logN_m)

# Display fitted line parameters
print(f"Fitted line parameters:")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")

# Step 8: Save catalog to file
catalog_filename = "galaxy_catalog.txt"
phot_table.write(catalog_filename, format='ascii', overwrite=True)
print(f"Catalog saved to {catalog_filename}")

def plot_pixel_count_histogram(image_data, combined_mask, region_size=200, bins=30):
    """
    Plot a histogram of average pixel counts in non-overlapping regions with a Gaussian fit.

    Parameters:
    - image_data (2D numpy array): The image data.
    - combined_mask (2D boolean array): The combined mask.
    - region_size (int): Size of the square regions in pixels.
    - bins (int): Number of histogram bins.
    """
    rows, cols = image_data.shape
    num_regions_row = rows // region_size
    num_regions_col = cols // region_size
    average_counts = []

    for i in range(num_regions_row):
        for j in range(num_regions_col):
            row_start = i * region_size
            row_end = row_start + region_size
            col_start = j * region_size
            col_end = col_start + region_size
            region = image_data[row_start:row_end, col_start:col_end]
            region_mask = combined_mask[row_start:row_end, col_start:col_end]
            if np.any(~region_mask):
                avg = np.mean(region[~region_mask])
                average_counts.append(avg)

    average_counts = np.array(average_counts)
    counts_hist, bins_hist = np.histogram(average_counts, bins=bins, density=True)
    bin_centers = (bins_hist[:-1] + bins_hist[1:]) / 2
    mu, std = norm.fit(average_counts)

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(average_counts, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black', label='Pixel Count Histogram')

    # Plot the Gaussian fit
    x_fit = np.linspace(bins_hist[0], bins_hist[-1], 1000)
    p_fit = norm.pdf(x_fit, mu, std)
    plt.plot(x_fit, p_fit, 'k', linewidth=2, label=f'Gaussian Fit: μ={mu:.2f}, σ={std:.2f}')

    plt.xlabel('Average Pixel Counts')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Average Pixel Counts in 200x200 Regions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.segmentation import detect_sources, SourceCatalog
# Load the FITS file
filepath = "/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits"  # Replace with your FITS file path
with fits.open(filepath) as hdul:
    image_data = hdul[0].data
    header = hdul[0].header

# Read calibration values from the FITS header
MAGZPT = header.get('MAGZPT', 25.0)  # Instrumental zero point (default 25.0)
MAGZRR = header.get('MAGZRR', 0.02)  # Error on zero point (default 0.02)

# Step 1: Mask edges and bright pixels
masked_image = np.copy(image_data).astype(float)  # Ensure it's float for masking
edge_margin = 20  # Smaller margin for edge masking
masked_image[:edge_margin, :] = np.nan  # Mask top edge
masked_image[-edge_margin:, :] = np.nan  # Mask bottom edge
masked_image[:, :edge_margin] = np.nan  # Mask left edge
masked_image[:, -edge_margin:] = np.nan  # Mask right edge
masked_image[masked_image > 50000] = np.nan  # Mask very bright pixels (less aggressive)

# Step 2: Replace NaNs with median for processing
median = np.nanmedian(masked_image)  # Median excluding NaN values
clean_image = np.nan_to_num(masked_image, nan=median)

# Step 3: Plot histogram of pixel counts after masking
non_nan_pixels = clean_image[clean_image > 0]  # Ignore zeros
plt.figure(figsize=(8, 6))
plt.hist(non_nan_pixels.flatten(), bins=100, color='blue', alpha=0.7)
plt.title('Histogram of Pixel Counts After Masking')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Step 4: Compute background statistics
mean, median, std = sigma_clipped_stats(clean_image, sigma=3.0)
print(f"Background Mean: {mean}, Median: {median}, Std Dev: {std}")

# Step 5: Detect sources using DAOStarFinder
threshold = median + (1.5 * std)  # Lower threshold for fainter sources
daofind = DAOStarFinder(threshold=threshold, fwhm=3.0)
sources = daofind(clean_image)

if sources is None or len(sources) == 0:
    print("No sources were detected.")
    exit()

print(f"Number of sources detected: {len(sources)}")

# Step 6: Perform aperture photometry
aperture_radius = 6
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=aperture_radius)
phot_table = aperture_photometry(clean_image, apertures)

# Convert counts to magnitudes
phot_table['mag'] = MAGZPT - 2.5 * np.log10(phot_table['aperture_sum'])

# Step 6: Compute cumulative counts for logN(m)
mag_bins = np.arange(np.min(mags), np.max(mags), 0.1)  # Smaller bin size
N_m = np.array([np.sum(mags <= m) for m in mag_bins])

# Handle zero counts
valid_bins = N_m > 0  # Keep only bins with sources
logN_m = np.log10(N_m[valid_bins])
mag_bins = mag_bins[valid_bins]

# Fit a line to logN(m)
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(mag_bins, logN_m)

# Plot logN(m) vs m with fitted line
plt.figure(figsize=(8, 6))
plt.scatter(mag_bins, logN_m, label='Measured Data', color='blue', alpha=0.7)
plt.plot(mag_bins, slope * mag_bins + intercept, '--', label=f'Fitted Line: logN(m) = {slope:.2f}m + {intercept:.2f}', color='red')
plt.xlabel('Magnitude (m)')
plt.ylabel('log N(<m)')
plt.title('Galaxy Number Counts')
plt.legend()
plt.grid()
plt.show()

# Display fitted line parameters
print(f"Fitted line parameters:")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")

# Fit a line to logN(m) data
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(mag_bins, logN_m)

# Display fitted line parameters
print(f"Fitted line parameters:")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")

# Step 9: Save catalog to file
catalog_filename = "galaxy_catalog.txt"
phot_table.write(catalog_filename, format='ascii', overwrite=True)
print(f"Catalog saved to {catalog_filename}")

# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.stats import sigma_clipped_stats
# from photutils.segmentation import detect_sources, SourceCatalog
# from photutils.aperture import CircularAperture, aperture_photometry

# # Load the FITS file
# filepath = "/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits"  # Replace with your FITS file path
# with fits.open(filepath) as hdul:
#     image_data = hdul[0].data
#     header = hdul[0].header

# # Read calibration values from the FITS header
# MAGZPT = header.get('MAGZPT', 25.0)  # Instrumental zero point (default 25.0)
# MAGZRR = header.get('MAGZRR', 0.02)  # Error on zero point (default 0.02)

# # Step 1: Compute background statistics
# mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)

# # Step 2: Detect sources using segmentation
# threshold = median + (5 * std)  # Detection threshold
# segmentation_map = detect_sources(image_data, threshold, npixels=5)  # Minimum size = 5 pixels

# if segmentation_map is None:
#     print("No sources were detected.")
#     exit()

# # Step 3: Perform photometry
# catalog = SourceCatalog(image_data, segmentation_map)

# # Define aperture radius (in pixels)
# aperture_radius = 6
# phot_table = []
# for source in catalog:
#     x, y = source.xcentroid, source.ycentroid
#     aperture = CircularAperture((x, y), r=aperture_radius)
#     photometry = aperture_photometry(image_data, aperture)
    
#     # Extract total counts within the aperture
#     total_counts = photometry['aperture_sum'][0]
    
#     # Compute calibrated magnitude
#     if total_counts > 0:
#         mag = MAGZPT - 2.5 * np.log10(total_counts)
#         phot_table.append({
#             'x': x,
#             'y': y,
#             'counts': total_counts,
#             'mag': mag
#         })

# # Convert to NumPy array for analysis
# phot_table = np.array(phot_table)

# # Step 4: Compute cumulative counts for logN(m)
# mags = np.array([entry['mag'] for entry in phot_table])  # Extract magnitudes into a NumPy array
# mag_bins = np.arange(np.min(mags), np.max(mags), 0.5)  # Magnitude bins
# N_m = [np.sum(mags <= m) for m in mag_bins]  # Cumulative number of sources
# logN_m = np.log10(N_m)

# # Step 5: Plot logN(m) vs m
# plt.figure(figsize=(8, 6))
# plt.plot(mag_bins, logN_m, label='Measured Data')
# plt.plot(mag_bins, 0.6 * mag_bins + np.mean(logN_m - 0.6 * mag_bins), '--', label='Theoretical (0.6m + const)')
# plt.xlabel('Magnitude (m)')
# plt.ylabel('log N(<m)')
# plt.title('Galaxy Number Counts')
# plt.legend()
# plt.grid()
# plt.show()

# # Step 6: Save catalog to file
# catalog_filename = "galaxy_catalog.txt"

# # Convert phot_table to a 2D array for saving
# phot_table_array = np.array([[entry['x'], entry['y'], entry['counts'], entry['mag']] for entry in phot_table])

# # Save the catalog to a file
# np.savetxt(catalog_filename, phot_table_array, fmt="%-10.4f", header="x y counts mag")
# print(f"Catalog saved to {catalog_filename}")

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
from sklearn.cluster import DBSCAN
import os
import warnings
from astropy.utils.metadata import MergeConflictWarning
from scipy.ndimage import median_filter

# Suppress specific warnings
warnings.simplefilter('ignore', MergeConflictWarning)
fits_file = '/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits'  # Replace with your FITS file path

with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

def display_fits(data, annotations=None):
    """
    Display function that includes data and annotations.
    Displays FITS image data using ZScale normalization and optional overlays.
    """
    # Calculate the median of non-NaN values
    median_value = np.nanmedian(data)
    # Replace NaN values with the median for display purposes
    display_data = np.where(np.isnan(data), median_value, data)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(display_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    plt.figure(figsize=(10, 8))
    im = plt.imshow(display_data, cmap='cool', norm=norm)
    plt.colorbar(im, label='Original Data Intensity')

    if annotations:
        for x, y in annotations:
            plt.plot(x, y, 'o', markersize=2, alpha=0.8, color='yellow')  # Yellow markers for detected sources

    plt.title('FITS Image with ZScale Stretch and Annotations')
    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.show()

display_fits(data)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter

# Load the FITS file
fits_file = '/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits'
ex = fits.open(fits_file)
data = ex[0].data

# Extract the section of interest
section1 = data[2997:3397, 1241:1641]

# Normalize the section to [0, 255] for thresholding
image_data_norm = (section1 - np.min(section1)) / (np.max(section1) - np.min(section1)) * 255
image_data_norm = image_data_norm.astype(np.uint8)

# Apply a binary threshold to isolate bright objects
threshold_value = 60
thresholded = np.where(image_data_norm > threshold_value, 255, 0).astype(np.uint8)

# Smooth the mask using a Gaussian filter
smoothed_mask = gaussian_filter(thresholded.astype(float), sigma=1.5)  # Adjust sigma for more or less smoothing

# Create a new masked image
masked_image = image_data_norm.copy()

# Apply the smoothed mask: Scale the mask between 0 and 1, then scale pixel values
smoothed_mask_normalized = smoothed_mask / np.max(smoothed_mask)  # Normalize mask to [0, 1]
masked_image = (masked_image * (1 - smoothed_mask_normalized)).astype(np.uint8)

# Plot the original section, thresholded image, smoothed mask, and final masked result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

# Original section
ax1.imshow(image_data_norm, cmap='gray', origin='lower')
ax1.set_title('Original Section')

# Thresholded image
ax2.imshow(thresholded, cmap='gray', origin='lower')
ax2.set_title('Thresholded Image')

# Smoothed mask
ax3.imshow(smoothed_mask, cmap='gray', origin='lower')
ax3.set_title('Smoothed Mask')

# Masked image with smoothed mask applied
ax4.imshow(masked_image, cmap='gray', origin='lower')
ax4.set_title('Masked Image with Smoothed Mask')

plt.tight_layout()
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.stats import sigma_clipped_stats
# from photutils.segmentation import detect_sources, SourceCatalog
# from photutils.background import Background2D, MedianBackground
# from scipy.ndimage import binary_dilation, binary_erosion

# # Load the FITS file
# fits_file = '/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits'  # Replace with your FITS file path
# with fits.open(fits_file) as hdul:
#     data = hdul[0].data  # CCD image data

# # Step 1: Mask edges and extremely bright pixels
# mask = np.zeros_like(data, dtype=bool)
# edge_margin = 50  # Exclude 25-pixel borders
# mask[:edge_margin, :] = True  # Mask top edge
# mask[-edge_margin:, :] = True  # Mask bottom edge
# mask[:, :edge_margin] = True  # Mask left edge
# mask[:, -edge_margin:] = True  # Mask right edge
# mask[data > 5000] = True  # Mask extremely bright pixels (e.g., stars or artifacts)

# # Step 2: Estimate background and subtract
# bkg = Background2D(data, (50, 50), filter_size=(3, 3), mask=mask, bkg_estimator=MedianBackground())
# data_subtracted = data - bkg.background

# # Step 3: Compute background statistics for source detection
# mean, median, std = sigma_clipped_stats(data_subtracted, sigma=3.0)

# # Step 4: Source detection using segmentation
# threshold = median + (5 * std)  # Detection threshold (adjust multiplier for sensitivity)
# segmentation_map = detect_sources(data_subtracted, threshold, npixels=5)  # Minimum 5 connected pixels

# # Step 5: Measure source properties
# if segmentation_map is None:
#     print("No sources detected.")
# else:
#     catalog = SourceCatalog(data_subtracted, segmentation_map)

#     # Count the total number of detected sources
#     num_sources = len(catalog)
#     print(f"Total number of detected sources: {num_sources}")

#     # Step 6: Visualize detected sources
#     plt.figure(figsize=(10, 10))
#     plt.imshow(data_subtracted, cmap='gray', origin='lower', vmin=np.percentile(data_subtracted, 5), vmax=np.percentile(data_subtracted, 99))
#     plt.colorbar(label='Pixel Value')
#     plt.title('Detected Sources (Galaxies)')
#     plt.xlabel('X Pixel')
#     plt.ylabel('Y Pixel')

#     # Overlay detected sources
#     for source in catalog:
#         plt.plot(source.xcentroid, source.ycentroid, 'ro', markersize=2, alpha=0.8)  # Red markers for sources

#     plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.stats import sigma_clipped_stats
# from photutils.segmentation import detect_sources, SourceCatalog
# from photutils.background import Background2D, MedianBackground
# from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
# from scipy.stats import norm

# # Load the FITS file
# fits_file = '/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits'  # Replace with your FITS file path
# with fits.open(fits_file) as hdul:
#     data = hdul[0].data  # CCD image data

# # Step 1: Mask edges and extremely bright pixels
# mask = np.zeros_like(data, dtype=bool)
# edge_margin = 50  # Exclude 25-pixel borders
# mask[:edge_margin, :] = True  # Mask top edge
# mask[-edge_margin:, :] = True  # Mask bottom edge
# mask[:, :edge_margin] = True  # Mask left edge
# mask[:, -edge_margin:] = True  # Mask right edge
# mask[data > 5000] = True  # Mask extremely bright pixels (e.g., stars or artifacts)

# # Step 2: Estimate background and subtract
# bkg = Background2D(data, (50, 50), filter_size=(3, 3), mask=mask, bkg_estimator=MedianBackground())
# data_subtracted = data - bkg.background

# # Step 3: Compute background statistics for source detection
# mean, median, std = sigma_clipped_stats(data_subtracted, sigma=3.0)

# # Step 4: Source detection using segmentation
# threshold = median + (5 * std)  # Detection threshold (adjust multiplier for sensitivity)
# segmentation_map = detect_sources(data_subtracted, threshold, npixels=5)  # Minimum 5 connected pixels

# if segmentation_map is None:
#     print("No sources detected.")
# else:
#     # Step 5: Measure source properties
#     catalog = SourceCatalog(data_subtracted, segmentation_map)

#     # Perform aperture photometry on detected sources
#     positions = np.array([(source.xcentroid, source.ycentroid) for source in catalog])
#     aperture_radius = 5  # Radius of the circular aperture (in pixels)
#     annulus_r_in = 7     # Inner radius of the background annulus
#     annulus_r_out = 10   # Outer radius of the background annulus

#     # Define circular apertures and annuli
#     apertures = CircularAperture(positions, r=aperture_radius)
#     annuli = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)

#     # Perform aperture photometry
#     phot_table = aperture_photometry(data_subtracted, apertures)
#     bkg_table = aperture_photometry(data_subtracted, annuli)

#     # Estimate background per pixel in the annulus
#     annulus_areas = annuli.area  # Area of the annuli
#     bkg_per_pixel = bkg_table['aperture_sum'] / annulus_areas

#     # Subtract the background contribution from the aperture flux
#     aperture_areas = apertures.area
#     phot_table['bkg_sum'] = bkg_per_pixel * aperture_areas  # Total background in the aperture
#     phot_table['net_flux'] = phot_table['aperture_sum'] - phot_table['bkg_sum']

#     # Filter non-finite and non-positive flux values
# fluxes = phot_table['net_flux'].data
# valid_fluxes = fluxes[np.isfinite(fluxes) & (fluxes > 0)]  # Exclude NaN, Inf, and non-positive fluxes

# # Step 6: Plot a histogram of frequency vs. counts
# if len(valid_fluxes) > 0:
#     plt.figure(figsize=(10, 6))
#     bins = np.logspace(np.log10(np.min(valid_fluxes)), np.log10(np.max(valid_fluxes)), 50)
#     hist, bin_edges, _ = plt.hist(valid_fluxes, bins=bins, color='gray', alpha=0.7, label='Counts Histogram')
#     plt.xscale('log')
#     plt.xlabel('Counts (Flux)')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Counts')

#     # Fit a Gaussian to the histogram
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
#     params = norm.fit(np.log10(valid_fluxes))  # Fit Gaussian in log space
#     gaussian_fit = norm.pdf(np.log10(bin_centers), loc=params[0], scale=params[1]) * len(valid_fluxes) * np.diff(bin_edges)[0]
#     plt.plot(bin_centers, gaussian_fit, 'r-', label='Gaussian Fit')

#     plt.legend()
#     plt.show()

#     # Step 7: Compute and plot log(N(m)) vs. m
#     magnitudes = -2.5 * np.log10(valid_fluxes)  # Convert flux to magnitudes (m = -2.5 * log10(flux))
#     sorted_magnitudes = np.sort(magnitudes)  # Sort magnitudes
#     cumulative_counts = np.arange(1, len(sorted_magnitudes) + 1)  # Cumulative counts

#     plt.figure(figsize=(10, 6))
#     plt.plot(sorted_magnitudes, np.log10(cumulative_counts), 'bo-', markersize=3, label='log(N(m)) vs. m')
#     plt.xlabel('Magnitude (m)')
#     plt.ylabel('log(N(m))')
#     plt.title('Cumulative Number Counts')
#     plt.grid()
#     plt.legend()
#     plt.show()
# else:
#     print("No valid fluxes available for analysis.")
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.stats import sigma_clipped_stats
# from photutils.segmentation import detect_sources, SourceCatalog
# from photutils.background import Background2D, MedianBackground
# from photutils.aperture import CircularAperture

# # Load the FITS file
# fits_file = '/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits'  # Replace with your FITS file path
# with fits.open(fits_file) as hdul:
#     data = hdul[0].data  # CCD image data

# # Step 1: Mask edges and extremely bright pixels
# mask = np.zeros_like(data, dtype=bool)
# edge_margin = 85  # Exclude 25-pixel borders
# mask[:edge_margin, :] = True  # Mask top edge
# mask[-edge_margin:, :] = True  # Mask bottom edge
# mask[:, :edge_margin] = True  # Mask left edge
# mask[:, -edge_margin:] = True  # Mask right edge
# mask[data > 5000] = True  # Mask extremely bright pixels (e.g., stars or artifacts)

# # Step 2: Estimate background and subtract
# # bkg = Background2D(data, (50, 50), filter_size=(3, 3), mask=mask, bkg_estimator=MedianBackground())
# # data_subtracted = data - bkg.background

# # Step 3: Compute background statistics for source detection
# mean, median, std = sigma_clipped_stats(data, sigma=3.0)

# # Step 4: Source detection using segmentation, applying the mask
# threshold = median + (5 * std)  # Detection threshold (adjust multiplier for sensitivity)
# segmentation_map = detect_sources(data, threshold, npixels=5, mask=mask)  # Apply mask during detection


# if segmentation_map is None:
#     print("No sources detected.")
# else:
#     # Step 5: Measure source properties
#     catalog = SourceCatalog(data, segmentation_map)
#     num_sources = len(catalog)
#     # print(f"Total number of detected sources: {num_sources}")

#     # Filter sources to exclude any that fall in masked regions
#     positions = []
#     for source in catalog:
#         x, y = source.xcentroid, source.ycentroid
#         if not mask[int(y), int(x)]:  # Include only sources not in the mask
#             positions.append((x, y))
#     positions = np.array(positions)

#     # Visualize the original data with masking and detected sources
#     plt.figure(figsize=(12, 10))

#     # Display the original image with masked regions
#     masked_image = np.where(mask, np.nan, data)  # Set masked regions to NaN for visualization
#     img = plt.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5), vmax=np.percentile(data, 99))
#     plt.imshow(mask, cmap='gray', origin='lower')
#     cbar = plt.colorbar(img, label='Pixel Intensity')
#     plt.title('Detected Sources with Masking')
#     plt.xlabel('X Pixel')
#     plt.ylabel('Y Pixel')

#     # Overlay detected sources

#     apertures = CircularAperture(positions, r=6)  # Adjust aperture radius as needed
#     # aperture_radius = 6  # pixels (3” diameter as per your specification)
#     # annulus_inner_radius = 8  # pixels
#     # annulus_outer_radius = 12  # pixels
#     apertures.plot(color='blue', lw=1.5, alpha=0.7)  # circles for detected sources

#     plt.show()

#     print(f"Total detected sources: {len(positions)}")

# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.stats import sigma_clipped_stats
# from photutils.detection import DAOStarFinder
# from photutils.aperture import CircularAperture
# from photutils.background import Background2D, MedianBackground

# # Load the FITS file
# fits_file = '/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits'  # Replace with your FITS file path
# with fits.open(fits_file) as hdul:
#     data = hdul[0].data  # CCD image data

# # Step 1: Mask edges and bright pixels
# mask = np.zeros_like(data, dtype=bool)
# edge_margin = 85
# mask[:edge_margin, :] = True  # Mask top edge
# mask[-edge_margin:, :] = True  # Mask bottom edge
# mask[:, :edge_margin] = True  # Mask left edge
# mask[:, -edge_margin:] = True  # Mask right edge
# mask[data > 5000] = True  # Mask extremely bright pixels

# # Step 2: Apply the mask to the image
# masked_image = np.where(mask, np.nan, data)  # Set masked regions to NaN

# # Step 3: Subtract the background
# bkg = Background2D(masked_image, (50, 50), filter_size=(3, 3), mask=mask, bkg_estimator=MedianBackground())
# masked_image_subtracted = masked_image - bkg.background

# # Step 4: Compute background statistics for DAOStarFinder
# mean, median, std = sigma_clipped_stats(masked_image_subtracted, sigma=3.0)

# # Step 5: Apply DAOStarFinder
# threshold = median + (5 * std)  # Detection threshold
# fwhm = 3.0  # Full Width at Half Maximum of the stars
# daofind = DAOStarFinder(threshold=threshold, fwhm=fwhm)

# # Detect sources in the masked and background-subtracted image
# sources = daofind(masked_image_subtracted)

# # Step 6: Filter invalid detections
# if sources is not None:
#     sources = sources.to_pandas()  # Convert to Pandas DataFrame for easier handling
#     sources = sources[sources['sharpness'] < 1.0]  # Example filter on sharpness
# else:
#     print("No sources detected.")

# # Step 7: Visualize detected sources
# if len(sources) > 0:
#     positions = np.transpose((sources['xcentroid'], sources['ycentroid']))  # Source positions
#     apertures = CircularAperture(positions, r=5)  # Define apertures for visualization

#     plt.figure(figsize=(12, 10))
#     plt.imshow(masked_image, cmap='gray', origin='lower', vmin=np.percentile(masked_image, 5), vmax=np.percentile(masked_image, 99))
#     # apertures.plot(color='red', lw=1.5, alpha=0.7)  # Overlay detected sources
#     plt.colorbar(label='Pixel Value')
#     plt.title('Detected Sources (DAOStarFinder) on Masked Image')
#     plt.xlabel('X Pixel')
#     plt.ylabel('Y Pixel')
#     plt.show()

#     print(sources.head())  # Display the first few rows of the detected sources
# else:
#     print("No valid sources detected.")

# print(f"Total detected sources: {len(sources)}")
