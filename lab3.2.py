import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.detection import DAOStarFinder
from photutils.segmentation import SegmentationImage, detect_threshold
from astropy.modeling.models import Gaussian2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from scipy.ndimage import binary_dilation, generate_binary_structure, center_of_mass
from collections import deque
import warnings
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress, norm
from skimage.restoration import denoise_wavelet
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
# ---------------------------
# Suppress Warnings
# ---------------------------
warnings.filterwarnings('ignore')

# ---------------------------
# Constants
# ---------------------------
FITS_FILE_PATH = '/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits'  # Update with your FITS file path
ZP_INST = 25.3  # Zero Point for magnitude calibration
ZP_ERR = 0.02   # Zero Point error
PIXEL_SCALE = 0.258  # arcseconds/pixel (default if not in FITS header)

# ---------------------------
# Function Definitions
# ---------------------------

def read_fits_image(fits_file_path):
    """
    Reads a FITS file and returns the image data and header.
    
    Parameters:
    - fits_file_path (str): Path to the FITS file.
    
    Returns:
    - image_data (2D NumPy array): The image data.
    - header (FITS Header): The header information.
    """
    with fits.open(fits_file_path) as hdul:
        hdul.info()  # Display FITS file structure (optional)
        image_data = hdul[0].data.astype(float)  # Ensure data is in float format
        header = hdul[0].header
    return image_data, header

def mask_edges(image_shape, boundary_width):
    """
    Creates a mask for the edges of the image.
    
    Parameters:
    - image_shape (tuple): Shape of the image (rows, cols).
    - boundary_width (int): Width of the boundary to mask (in pixels).
    
    Returns:
    - edge_mask (2D boolean array): Mask with True for edge pixels.
    """
    edge_mask = np.zeros(image_shape, dtype=bool)
    rows, cols = image_shape
    # Top and Bottom edges
    edge_mask[:boundary_width, :] = True
    edge_mask[-boundary_width:, :] = True
    # Left and Right edges
    edge_mask[:, :boundary_width] = True
    edge_mask[:, -boundary_width:] = True
    return edge_mask

def mask_bright_regions(image_data, brightness_threshold):
    """
    Creates a mask for very bright (saturated) pixels.
    
    Parameters:
    - image_data (2D NumPy array): The image data.
    - brightness_threshold (float): Pixel intensity above which pixels are considered bright.
    
    Returns:
    - bright_mask (2D boolean array): Mask with True for bright pixels.
    """
    bright_mask = image_data > brightness_threshold
    return bright_mask

def create_combined_mask(image_data, boundary_width=85, brightness_threshold=50000, dilation_iterations=3):
    """
    Creates a combined mask that includes edges and very bright regions, then expands the mask.
    
    Parameters:
    - image_data (2D NumPy array): The image data.
    - boundary_width (int): Width of the boundary to mask (in pixels).
    - brightness_threshold (float): Intensity above which pixels are considered bright.
    - dilation_iterations (int): Number of dilation iterations to expand the mask.
    
    Returns:
    - combined_mask (2D boolean array): The combined and expanded mask.
    """
    # Mask the edges
    edge_mask = mask_edges(image_data.shape, boundary_width)
    print("Edges masked.")
    
    # Mask very bright regions
    bright_mask = mask_bright_regions(image_data, brightness_threshold)
    print(f"Very bright regions (pixels > {brightness_threshold}) masked.")
    
    # Combine masks
    combined_mask = edge_mask | bright_mask
    
    # Expand the mask using dilation
    structure = generate_binary_structure(2, 2)  # 8-connected
    combined_mask = binary_dilation(combined_mask, structure=structure, iterations=dilation_iterations)
    print(f"Combined mask expanded with {dilation_iterations} dilation iterations.")
    
    return combined_mask

def visualize_mask(image_data, mask, title='Combined Mask'):
    """
    Visualizes the image data with the mask overlayed.
    
    Parameters:
    - image_data (2D NumPy array): The image data.
    - mask (2D boolean array): The mask to overlay.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    norm_orig = simple_norm(image_data, 'sqrt', percent=99)
    plt.imshow(image_data, norm=norm_orig, cmap='gray', origin='upper')
    plt.colorbar(label='Pixel Intensity')
    plt.title('Original CCD Image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    
    # Masked Image
    plt.subplot(1, 2, 2)
    masked_image = np.copy(image_data)
    masked_image[mask] = np.nan  # Set masked pixels to NaN for transparency
    
    norm_masked = simple_norm(masked_image, 'sqrt', percent=99)
    plt.imshow(masked_image, norm=norm_masked, cmap='gray', origin='upper')
    plt.colorbar(label='Pixel Intensity')
    plt.title(title)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    
    # Overlay the mask with transparency
    masked_overlay = np.ma.masked_where(~mask, mask)
    plt.imshow(masked_overlay, cmap='plasma', alpha=0.5, origin='upper')  # Updated colormap and transparency
    
    plt.tight_layout()
    plt.show()

def save_mask(mask, filename='combined_mask.npy'):
    """
    Saves the mask to a NumPy binary file.
    
    Parameters:
    - mask (2D boolean array): The mask to save.
    - filename (str): Filename for the saved mask.
    """
    np.save(filename, mask)
    print(f"Mask saved to '{filename}'.")

def load_mask(filename='combined_mask.npy'):
    """
    Loads a mask from a NumPy binary file.
    
    Parameters:
    - filename (str): Filename of the saved mask.
    
    Returns:
    - mask (2D boolean array): The loaded mask.
    """
    mask = np.load(filename)
    print(f"Mask loaded from '{filename}'.")
    return mask

def adaptive_brightness_threshold(image_data, percentile=99.0):
    """
    Calculates an adaptive brightness threshold based on a specified percentile.
    
    Parameters:
    - image_data (2D NumPy array): The image data.
    - percentile (float): The percentile to determine the threshold.
    
    Returns:
    - threshold (float): Adaptive brightness threshold.
    """
    threshold = np.percentile(image_data, percentile)
    return threshold

def threshold_image(image, threshold_value, mask):
    """
    Threshold the image to create a binary mask.
    
    Parameters:
    - image (2D NumPy array): The image data.
    - threshold_value (float): Pixel value threshold.
    - mask (2D boolean array): Existing mask to exclude regions.
    
    Returns:
    - binary_image (2D boolean array): Binary mask where True indicates potential sources.
    """
    binary_image = (image > threshold_value) & (~mask)
    return binary_image

def perform_photometry_with_daofinder(image_data, mask, fwhm=3.0, threshold_sigma=5.0, aperture_radius=6, annulus_r_in=10, annulus_r_out=12):
    """
    Detect sources using DAOStarFinder and perform aperture photometry.
    
    Parameters:
    - image_data (2D NumPy array): The image data.
    - mask (2D boolean array): The combined mask.
    - fwhm (float): FWHM for DAOStarFinder.
    - threshold_sigma (float): Detection threshold in sigma.
    - aperture_radius (float): Radius of the circular aperture.
    - annulus_r_in (float): Inner radius of the background annulus.
    - annulus_r_out (float): Outer radius of the background annulus.
    
    Returns:
    - phot_df (Pandas DataFrame): Photometry results.
    """
    # Estimate background statistics
    mean, median, std = sigma_clipped_stats(image_data, sigma=5.0)
    
    # Initialize DAOStarFinder
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma*std)
    
    # Detect sources in the masked image
    sources = daofind(image_data - median)
    
    if sources is None:
        print("No sources detected by DAOStarFinder.")
        return pd.DataFrame()
    
    print(f"Number of sources detected by DAOStarFinder: {len(sources)}")
    
    # Perform aperture photometry
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=aperture_radius)
    annuli = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)
    
    apertures_masks = apertures.to_mask(method='exact')
    annuli_masks = annuli.to_mask(method='exact')
    
    phot_table = aperture_photometry(image_data, apertures)
    
    # Background subtraction
    bkg_median = []
    for i in range(len(annuli)):
        annulus_data = annuli_masks[i].multiply(image_data)
        annulus_data_1d = annulus_data.flatten()
        # Flatten the mask as well
        mask_flat = annuli_masks[i].data.flatten() > 0
        annulus_data_1d = annulus_data_1d[mask_flat]
        if len(annulus_data_1d) == 0:
            median_sigclip = 0.0
        else:
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d, sigma=3.0)
        bkg_median.append(median_sigclip)
    
    phot_table['bkg_median'] = bkg_median
    phot_table['bkg_total'] = phot_table['bkg_median'] * apertures.area
    phot_table['net_flux'] = phot_table['aperture_sum'] - phot_table['bkg_total']
    
    # Calculate uncertainties
    phot_table['sigma_flux'] = np.sqrt(phot_table['aperture_sum'] + phot_table['bkg_total'])
    
    # Convert flux to magnitude
    with np.errstate(divide='ignore'):
        phot_table['magnitude'] = -2.5 * np.log10(phot_table['net_flux']) + ZP_INST
    phot_table['magnitude_error'] = (2.5 / np.log(10)) * (phot_table['sigma_flux'] / phot_table['net_flux']) + ZP_ERR
    
    # Convert to DataFrame
    phot_df = phot_table.to_pandas()
    
    # Rename 'aperture_sum' to 'flux' for consistency
    phot_df.rename(columns={'aperture_sum': 'flux'}, inplace=True)
    
    return phot_df

def classify_sources_kmeans(phot_df, n_clusters=2):
    """
    Classify sources into stars and galaxies using K-Means clustering.
    
    Parameters:
    - phot_df (Pandas DataFrame): Photometry results.
    - n_clusters (int): Number of clusters for K-Means.
    
    Returns:
    - phot_df (Pandas DataFrame): Updated DataFrame with classification labels.
    - clusters_df (Pandas DataFrame): Cluster centers for analysis.
    """
    # Select features for clustering
    required_features = ['magnitude', 'flux', 'sigma_flux']
    available_features = [feat for feat in required_features if feat in phot_df.columns]
    
    if len(available_features) < len(required_features):
        print(f"Missing features for clustering: {set(required_features) - set(available_features)}")
        return phot_df, pd.DataFrame()
    
    # Prepare feature matrix
    features = phot_df[available_features].dropna()
    
    # Handle any infinite or NaN values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(features_scaled)
    
    # Assign cluster labels
    phot_df = phot_df.iloc[features.index].copy()
    phot_df['cluster'] = kmeans.labels_
    
    # Inverse transform cluster centers for interpretation
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    clusters_df = pd.DataFrame(cluster_centers, columns=available_features)
    clusters_df['cluster'] = range(n_clusters)
    print("Cluster Centers (Unscaled Features):")
    print(clusters_df)
    
    # Plotting K-Means clustering results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=phot_df, x='flux', y='sigma_flux', hue='cluster', palette='viridis', alpha=0.6)
    plt.title('K-Means Clustering: Flux vs. Sigma Flux')
    plt.xlabel('Flux')
    plt.ylabel('Sigma Flux')
    plt.legend(title='Cluster')
    plt.show()
    
    # Pairplot for additional visualization
    sns.pairplot(phot_df, vars=['magnitude', 'flux', 'sigma_flux'], hue='cluster', palette='viridis')
    plt.show()
    
    # Calculate Silhouette Score
    if n_clusters > 1:
        score = silhouette_score(features_scaled, kmeans.labels_)
        print(f"Silhouette Score for K={n_clusters}: {score:.2f}")
    else:
        print("Silhouette Score requires at least 2 clusters.")
    
    # Elbow Method to determine optimal K
    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans_temp.fit(features_scaled)
        inertia.append(kmeans_temp.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.xticks(K_range)
    plt.grid(True)
    plt.show()
    
    # Assign source types based on cluster properties
    # This is an assumption; adjust based on actual cluster centers
    # Typically, stars are brighter (lower magnitude) and have higher flux
    # Galaxies are fainter and have lower flux
    if n_clusters == 2:
        # Determine which cluster has higher flux (stars)
        star_cluster = clusters_df.loc[clusters_df['flux'].idxmax(), 'cluster']
        phot_df['source_type'] = phot_df['cluster'].map({star_cluster: 'Star'})
        # Assign 'Galaxy' to the other cluster
        phot_df['source_type'].fillna('Galaxy', inplace=True)
    else:
        phot_df['source_type'] = phot_df['cluster'].astype(str)
    
    print("Source Classification Completed.")
    print(phot_df[['id', 'flux', 'magnitude', 'sigma_flux', 'cluster', 'source_type']].head())
    
    return phot_df, clusters_df


def classify_sources_dbscan(phot_df, eps=0.5, min_samples=5):
    """
    Classify sources into stars and galaxies using DBSCAN clustering.
    
    Parameters:
    - phot_df (Pandas DataFrame): Photometry results.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    - phot_df_clean (Pandas DataFrame): Updated DataFrame with DBSCAN cluster labels and source types.
    - dbscan_clusters_df (Pandas DataFrame): Empty DataFrame (DBSCAN does not provide cluster centers).
    """

    # ---------------------------
    # 1. Select Features for Clustering
    # ---------------------------
    required_features = ['magnitude', 'flux', 'sigma_flux']  
    available_features = [feat for feat in required_features if feat in phot_df.columns]
    
    if len(available_features) < len(required_features):
        missing = set(required_features) - set(available_features)
        print(f"Missing features for DBSCAN clustering: {missing}")
        return phot_df, pd.DataFrame()
    
    # ---------------------------
    # 2. Prepare Feature Matrix
    # ---------------------------
    features = phot_df[available_features].dropna()
    
    # Handle any infinite or NaN values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    
    if features.empty:
        print("No valid features available for DBSCAN clustering after cleaning.")
        return phot_df, pd.DataFrame()
    
    # ---------------------------
    # 3. Scale Features
    # ---------------------------
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # ---------------------------
    # 4. Initialize and Fit DBSCAN
    # ---------------------------
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(features_scaled)
    
    # ---------------------------
    # 5. Assign Cluster Labels
    # ---------------------------
    # Create a copy of phot_df to avoid SettingWithCopyWarning
    phot_df_clean = phot_df.loc[features.index].copy()
    phot_df_clean['cluster_dbscan'] = dbscan.labels_
    
    # ---------------------------
    # 6. Plotting DBSCAN Clustering Results
    # ---------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=phot_df_clean,
        x='flux',
        y='sigma_flux',
        hue='cluster_dbscan',
        palette='viridis',
        alpha=0.6,
        legend='full'
    )
    plt.title('DBSCAN Clustering: Flux vs. Sigma Flux')
    plt.xlabel('Flux')
    plt.ylabel('Sigma Flux')
    plt.legend(title='Cluster DBSCAN', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Pairplot for Additional Visualization (Optional)
    sns.pairplot(
        phot_df_clean,
        vars=['magnitude', 'flux', 'sigma_flux'],
        hue='cluster_dbscan',
        palette='viridis'
    )
    plt.show()
    
    # ---------------------------
    # 7. Calculate Silhouette Score
    # ---------------------------
    unique_labels = set(dbscan.labels_)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    if n_clusters > 1:
        # Exclude noise points (-1) for silhouette score
        mask = dbscan.labels_ != -1
        if np.sum(mask) > 1:
            score = silhouette_score(features_scaled[mask], dbscan.labels_[mask])
            print(f"Silhouette Score for DBSCAN (excluding noise): {score:.2f}")
        else:
            print("Not enough non-noise points to calculate Silhouette Score for DBSCAN.")
    else:
        print("Silhouette Score requires at least 2 clusters (excluding noise).")
    
    # ---------------------------
    # 8. Assign Source Types Based on DBSCAN Clusters
    # ---------------------------
    if n_clusters >= 2:
        # Determine which cluster has higher flux (assumed to be 'Star')
        cluster_flux_means = phot_df_clean.groupby('cluster_dbscan')['flux'].mean()
        star_cluster = cluster_flux_means.idxmax()
        
        # Assign 'Star' to star_cluster, 'Galaxy' to others, 'Noise' to -1
        phot_df_clean['source_type_dbscan'] = phot_df_clean['cluster_dbscan'].apply(
            lambda x: 'Star' if x == star_cluster else ('Noise' if x == -1 else 'Galaxy')
        )
    else:
        # If less than 2 clusters, label all as 'Galaxy' or 'Noise'
        phot_df_clean['source_type_dbscan'] = phot_df_clean['cluster_dbscan'].apply(
            lambda x: 'Noise' if x == -1 else 'Galaxy'
        )
    
    # ---------------------------
    # 9. Final Output
    # ---------------------------
    print("DBSCAN Source Classification Completed.")
    print(phot_df_clean[['id', 'flux', 'magnitude', 'sigma_flux', 'cluster_dbscan', 'source_type_dbscan']].head())
    
    return phot_df_clean, pd.DataFrame()



def plot_k_distance(features_scaled, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(features_scaled)
    distances, indices = nbrs.kneighbors(features_scaled)
    distances = np.sort(distances[:, k-1], axis=0)
    plt.figure(figsize=(8,6))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.title(f'K-distance Graph for DBSCAN (k={k})')
    plt.grid(True)
    plt.show()

# Usage within classify_sources_dbscan before fitting DBSCAN:
    plot_k_distance(features_scaled, k=5)


def plot_number_count(phot_df, delta_m=0.5):
    """
    Plots the number counts (logN) vs. magnitude and fits a linear relation.
    
    Parameters:
    - phot_df (Pandas DataFrame): Photometry results with magnitudes.
    - delta_m (float): Bin size for magnitude.
    """
    phot_df_clean = phot_df.dropna(subset=['magnitude'])
    bins = np.arange(np.floor(phot_df_clean['magnitude'].min()),
                    np.ceil(phot_df_clean['magnitude'].max()) + delta_m,
                    delta_m)
    counts, bin_edges = np.histogram(phot_df_clean['magnitude'], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate logN(m)
    log_counts = np.log10(counts + 1)  # Add 1 to avoid log(0)
    
    # Perform linear regression using linregress
    slope, intercept, r_value, p_value, std_err = linregress(bin_centers, log_counts)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(bin_centers, log_counts, yerr=np.sqrt(counts)/counts, fmt='o', label='Observed Counts', ecolor='gray', capsize=3)
    plt.plot(bin_centers, slope * bin_centers + intercept, 'r--', label=f'Fit: logN(m) = {slope:.2f}m + {intercept:.2f}')
    plt.xlabel('Magnitude (m)')
    plt.ylabel(r'$\log_{10} N(m)$')
    plt.title('Number Counts vs. Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Fitted Line: logN(m) = {slope:.2f}m + {intercept:.2f}")
    print(f"Fit R-squared: {r_value**2:.3f}")

def plot_pixel_count_histogram(average_counts):
    """
    Plots a histogram of average pixel counts with a Gaussian fit.
    
    Parameters:
    - average_counts (1D NumPy array): Average pixel counts in regions.
    """
    # Remove NaN values if any
    average_counts = average_counts[~np.isnan(average_counts)]
    
    # Plot histogram
    counts_hist, bins_hist = np.histogram(average_counts, bins=30, density=True)
    bin_centers = (bins_hist[:-1] + bins_hist[1:]) / 2
    mu, std = norm.fit(average_counts)
    
    # Plot the Gaussian fit
    x_fit = np.linspace(bins_hist[0], bins_hist[-1], 1000)
    p_fit = norm.pdf(x_fit, mu, std)
    plt.figure(figsize=(8, 6))
    plt.hist(average_counts, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='Pixel Count Histogram')
    plt.plot(x_fit, p_fit, 'k', linewidth=2, label=f'Gaussian Fit: μ={mu:.2f}, σ={std:.2f}')
    plt.xlabel('Average Pixel Counts')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Average Pixel Counts in 200x200 Regions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Gaussian Fit Parameters:\nMean (μ): {mu:.2f}\nStandard Deviation (σ): {std:.2f}")

# def plot_original_masked(original_image, masked_image, combined_mask, region_start=400, region_end=900):
#     """
#     Visualizes a specific region of the original and masked images.
    
#     Parameters:
#     - original_image (2D NumPy array): The original image data.
#     - masked_image (2D NumPy array): The masked image data.
#     - combined_mask (2D boolean array): The combined mask.
#     - region_start (int): Starting pixel for the region.
#     - region_end (int): Ending pixel for the region.
#     """
#     xlim = (region_start, region_end)
#     ylim = (region_start, region_end)
#     original_region = original_image[ylim[0]:ylim[1], xlim[0]:xlim[1]]
#     masked_region = masked_image[ylim[0]:ylim[1], xlim[0]:xlim[1]]
    
#     plt.figure(figsize=(12, 6))
    
#     # Original Region with Apertures
#     plt.subplot(1, 2, 1)
#     norm_orig = simple_norm(original_region, 'sqrt', percent=99)
#     plt.imshow(original_region, norm=norm_orig, cmap='gray', origin='lower')
#     plt.colorbar(label='Pixel Counts')
#     plt.title('Original Region')
#     plt.xlabel('X Pixel')
#     plt.ylabel('Y Pixel')
    
#     # Masked Image
#     plt.subplot(1, 2, 2)
#     norm_masked = simple_norm(masked_region, 'sqrt', percent=99)
#     plt.imshow(masked_region, norm=norm_masked, cmap='gray', origin='lower')
#     plt.colorbar(label='Pixel Counts')
#     plt.title('Masked CCD Image')
#     plt.xlabel('X Pixel')
#     plt.ylabel('Y Pixel')
    
#     plt.tight_layout()
#     plt.show()

def save_catalogue(phot_df, filename='galaxy_catalogue.csv'):
    """
    Save the photometry catalogue to a CSV file.

    Parameters:
    - phot_df (pandas.DataFrame): DataFrame containing photometry results.
    - filename (str): Name of the CSV file to save.
    """
    catalogue_columns = ['id', 'xcentroid', 'ycentroid', 'flux',
                        'bkg_median', 'bkg_total', 'net_flux',
                        'sigma_flux', 'magnitude', 'magnitude_error',
                        'cluster', 'source_type']
    
    # Check if all required columns exist
    missing_columns = [col for col in catalogue_columns if col not in phot_df.columns]
    if missing_columns:
        print(f"Cannot save catalogue. Missing columns: {missing_columns}")
        return
    
    phot_df[catalogue_columns].to_csv(filename, index=False)
    print(f"Catalogue saved to '{filename}'.")

# ---------------------------
# Main Execution Functions
# ---------------------------

def main_mask_creation():
    """
    Creates and saves the combined mask for the image.
    """
    # Step 1: Read the FITS image
    image_data, header = read_fits_image(FITS_FILE_PATH)
    print("FITS image loaded.")
    
    # Step 2: Define masking parameters
    boundary_width = 85          # Width of the edge to mask (in pixels)
    brightness_threshold = 50000  # Pixel intensity threshold for bright regions
    dilation_iterations = 1      # Number of dilation iterations to expand the mask
    
    # Step 3: Create the combined mask with expanded coverage
    combined_mask = create_combined_mask(
        image_data,
        boundary_width=boundary_width,
        brightness_threshold=brightness_threshold,
        dilation_iterations=dilation_iterations
    )
    print("Combined mask created.")
    
    # Step 4: Visualize the original and masked images
    visualize_mask(image_data, combined_mask, title='Combined Mask (Edges & Overbright Regions)')
    
    # Step 5: Save the mask for future use
    save_mask(combined_mask, filename='combined_mask.npy')
    
    # Optional: Save the mask as a FITS file for compatibility
    hdu_mask = fits.PrimaryHDU(data=combined_mask.astype(np.uint8))  # Convert boolean to uint8 (0 and 1)
    hdu_mask.writeto('combined_mask.fits', overwrite=True)
    print("Mask also saved as 'combined_mask.fits'.")

def main_source_detection():
    """
    Performs source detection using DAOStarFinder, photometry, and classification using both K-Means and DBSCAN.
    Also includes various visualizations.
    """
    # Step 1: Read the FITS image
    image_data, header = read_fits_image(FITS_FILE_PATH)
    print("FITS image loaded.")
    
    # Step 2: Load the mask
    mask = load_mask('combined_mask.npy')
    
    # Step 3: Adaptive Thresholding
    # Lower the percentile from 99.9 to 99.0 or 98.0 to ensure source detection
    adaptive_threshold = adaptive_brightness_threshold(image_data, percentile=99.0)
    print(f"Adaptive brightness threshold (99.0 percentile): {adaptive_threshold:.2f}")
    
    # Step 4: Detect sources and perform photometry using DAOStarFinder
    phot_df = perform_photometry_with_daofinder(
        image_data,
        mask,
        fwhm=3.0,
        threshold_sigma=5.0,
        aperture_radius=6,
        annulus_r_in=10,
        annulus_r_out=15
    )
    
    if phot_df.empty:
        print("No photometry data available. Exiting source detection.")
        return
    
    print("Photometry Results (First 5 Entries):")
    print(phot_df.head())
    
    # Step 5: K-Means Clustering for Classification
    phot_df_classified_kmeans, clusters_df_kmeans = classify_sources_kmeans(phot_df, n_clusters=2)
    
    # Step 6: DBSCAN Clustering for Classification
    phot_df_classified_dbscan, clusters_df_dbscan = classify_sources_dbscan(phot_df, eps=0.5, min_samples=5)
    
    # Step 8: Plot Number Count with Error Bars (using K-Means classification)
    plot_number_count(phot_df_classified_kmeans)
    
    # Step 9: Plot Pixel Count Histogram with Gaussian Fit (Optional)
    # Define region size
    region_size = 200  # pixels
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
            region_mask = mask[row_start:row_end, col_start:col_end]
            if np.any(~region_mask):
                avg = np.mean(region[~region_mask])
                average_counts.append(avg)
    average_counts = np.array(average_counts)
    plot_pixel_count_histogram(average_counts)
    
    # Step 10: Visualize Original vs. Masked Images
    masked_image = np.copy(image_data)
    masked_image[mask] = np.nan  # Apply mask for visualization
    # plot_original_masked(image_data, masked_image, mask, region_start=400, region_end=900)
    
    # Step 11: Save the Photometry Catalogue with Classification Labels
    save_catalogue(phot_df_classified_kmeans, filename='galaxy_catalogue.csv')
    
    print("Source Detection and Classification Completed.")

def main_histogram():
    """
    Plots a histogram of unmasked pixel brightness with a Gaussian fit.
    """
    # Step 1: Load image and mask
    image_data, header = read_fits_image(FITS_FILE_PATH)
    mask = load_mask('combined_mask.npy')
    print("Image and mask loaded.")
    
    # Step 2: Extract unmasked pixel values
    unmasked_pixels = image_data[~mask]
    print(f"Number of unmasked pixels: {len(unmasked_pixels)}")
    
    # Step 3: Plot histogram with Gaussian fit
    plt.figure(figsize=(8, 6))
    counts, bins, patches = plt.hist(unmasked_pixels.flatten(), bins=100, density=True, alpha=0.6, color='g', edgecolor='black', label='Pixel Brightness Histogram')
    
    # Fit a Gaussian
    mu, std = norm.fit(unmasked_pixels.flatten())
    x = np.linspace(bins[0], bins[-1], 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=2, label=f'Gaussian Fit: μ={mu:.2f}, σ={std:.2f}')
    
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Unmasked Pixel Brightness with Gaussian Fit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(f"Gaussian Fit Parameters:\nMean (μ): {mu:.2f}\nStandard Deviation (σ): {std:.2f}")

# ---------------------------
# Execute the Main Function
# ---------------------------

def main():
    """
    Main function to execute the entire pipeline.
    """
    # Step 1: Create and save the combined mask
    print("=== Mask Creation ===")
    main_mask_creation()
    
    # Step 2: Perform source detection and analysis
    print("\n=== Source Detection and Analysis ===")
    main_source_detection()
    
    # Step 3: Generate Pixel Brightness Histogram
    print("\n=== Pixel Brightness Histogram ===")
    main_histogram()

if __name__ == "__main__":
    main()
