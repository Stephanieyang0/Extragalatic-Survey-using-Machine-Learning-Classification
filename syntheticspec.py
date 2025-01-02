import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synphot.models import BlackBodyNorm1D, GaussianFlux1D
from astropy.modeling import models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import chisquare, ks_2samp, mean_squared_error
from scipy.stats import norm, ks_2samp
import seaborn as sns
from astropy import units as u
plt.rcParams.update({
    'font.size': 13,             # Default font size
    'axes.titlesize': 15,        # Title font size
    'axes.labelsize': 14,        # Axis labels font size
    'legend.fontsize': 14,       # Legend labels font size
    'legend.title_fontsize': 14  # Legend title font size
})

# def generate_blackbody_flux(wavelengths, temperature):
#     """
#     Generate blackbody flux per unit wavelength using Astropy's BlackBody1D model.
    
#     Parameters:
#     - wavelengths (Quantity): Wavelengths with units (e.g., Angstrom)
#     - temperature (Quantity): Temperature with units (e.g., K)
    
#     Returns:
#     - flux (Quantity): Blackbody flux with units (e.g., erg / (s cm^2 Angstrom))
#     """
#     # Initialize the BlackBody1D model with the given temperature
#     blackbody = models.BlackBody(temperature=temperature)
    
#     # Compute the flux for the given wavelengths
#     flux = blackbody(wavelengths)
    
#     return flux

# def main():
#     try:
#         print("Defining wavelengths and temperature...")
#         # Define wavelengths from 3000 to 8000 Angstrom with 500 points
#         wavelengths = np.linspace(3000, 8000, 500) * u.Angstrom
        
#         # Define temperature (e.g., 6000 K)
#         temperature = 6000 * u.K
        
#         print("Generating synthetic flux...")
#         flux = generate_blackbody_flux(wavelengths, temperature)
        
#         print("Plotting the synthetic spectrum...")
#         plt.figure(figsize=(10, 6))
#         plt.plot(wavelengths, flux, label=f'Blackbody {temperature.value} K')
#         plt.xlabel('Wavelength (Angstrom)', fontsize=14)
#         plt.ylabel('Flux (erg / (s cm² Angstrom))', fontsize=14)
#         plt.title('Synthetic Blackbody Spectrum', fontsize=16)
#         plt.legend(fontsize=12)
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.tight_layout()
#         plt.show()
        
#         print("Saving synthetic spectrum to file...")
#         synthetic_spectrum = np.column_stack((wavelengths.value, flux.value))
#         filename = 'synthetic_blackbody_spectrum.txt'
#         np.savetxt(filename, synthetic_spectrum, 
#                    header='Wavelength(Angstrom) Flux(erg/s/cm^2/Angstrom)', fmt='%f %e')
#         print(f"Synthetic blackbody spectrum saved to '{filename}'.")
        
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#         main()


# # ------------------------------
# # Step 1: Load Observational Data
# # ------------------------------

# # Replace 'observational_catalog.csv' with your actual data file
# observational_catalog = pd.read_csv('galaxy_catalogue.csv')

# # Extract magnitudes
# observational_magnitudes = observational_catalog['magnitude']

# # Define magnitude bins
# mag_bins = np.arange(15, 30, 0.5)
# mag_bin_centers = (mag_bins[:-1] + mag_bins[1:]) / 2

# # Compute number counts
# observational_counts, _ = np.histogram(observational_magnitudes, bins=mag_bins)

# # ------------------------------
# # Step 2: Generate Synthetic Blackbody Simulation
# # ------------------------------


# wavelengths = np.linspace(3000, 8000, 500) * u.Angstrom
# temperatures = [5000, 6000, 7000, 8000] * u.K

# synthetic_fluxes = []

# for temp in temperatures:
#     flux = generate_blackbody_flux(wavelengths, temp)
#     synthetic_fluxes.append(flux.value)

# # Convert fluxes to AB magnitudes (simplistic conversion)
# def flux_to_ab_mag(flux, band_eff_wavelength):
#     flux_jy = (flux * 1e-23) / (band_eff_wavelength.to(u.cm).value)
#     mag = -2.5 * np.log10(flux_jy / 3631)
#     return mag

# band_eff_wavelength = 5500 * u.Angstrom
# synthetic_magnitudes = [flux_to_ab_mag(flux, band_eff_wavelength) for flux in synthetic_fluxes]

# # Combine all synthetic magnitudes
# synthetic_magnitudes_combined = np.concatenate(synthetic_magnitudes)

# # ------------------------------
# # Step 3: Apply Selection Effects
# # ------------------------------

# # Define magnitude limits
# mag_min = 15
# mag_max = 30

# # Filter synthetic magnitudes
# synthetic_mask = (synthetic_magnitudes_combined >= mag_min) & (synthetic_magnitudes_combined <= mag_max)
# synthetic_magnitudes_filtered = synthetic_magnitudes_combined[synthetic_mask]

# # Define a completeness function
# def completeness(mag):
#     if mag <= 25:
#         return 1.0
#     elif 25 < mag <= 30:
#         return (30 - mag) / 5
#     else:
#         return 0.0

# # Apply completeness
# completeness_factors = np.array([completeness(mag) for mag in synthetic_magnitudes_filtered])
# synthetic_magnitudes_complete = synthetic_magnitudes_filtered[np.random.rand(len(synthetic_magnitudes_filtered)) < completeness_factors]

# # ------------------------------
# # Step 4: Construct Number Counts
# # ------------------------------

# # Observational counts (already computed)
# # synthetic_counts_complete, _ = np.histogram(synthetic_magnitudes_complete, bins=mag_bins)

# # For demonstration, recompute synthetic counts after filtering
# synthetic_counts_complete, _ = np.histogram(synthetic_magnitudes_complete, bins=mag_bins)

# # ------------------------------
# # Step 5: Statistical Comparison
# # ------------------------------

# # Chi-Squared Test
# # epsilon = 1e-5
# # synthetic_counts_safe = synthetic_counts_complete.copy()
# # synthetic_counts_safe[synthetic_counts_safe == 0] = epsilon

# # chi_stat, p_value = chisquare(f_obs=observational_counts, f_exp=synthetic_counts_safe)
# # print(f"Chi-Squared Statistic: {chi_stat:.2f}")
# # print(f"P-Value: {p_value:.4f}")

# # # K-S Test
# # ks_stat, ks_p_value = ks_2samp(observational_magnitudes, synthetic_magnitudes_complete)
# # print(f"K-S Statistic: {ks_stat:.2f}")
# # print(f"K-S P-Value: {ks_p_value:.4f}")

# # # RMSE
# # rmse = np.sqrt(mean_squared_error(observational_counts, synthetic_counts_complete))
# # nrmse = rmse / np.mean(observational_counts)
# # print(f"RMSE: {rmse:.2f}")
# # print(f"NRMSE: {nrmse:.2f}")

# # ------------------------------
# # Step 6: Visualization
# # ------------------------------

# # Overlayed Histograms
# plt.figure(figsize=(10, 6))
# plt.bar(mag_bin_centers - 0.15, observational_counts, width=0.3, color='blue', alpha=0.7, label='Observational Data')
# plt.bar(mag_bin_centers + 0.15, synthetic_counts_complete, width=0.3, color='red', alpha=0.7, label='Synthetic Simulation')
# plt.xlabel('Magnitude', fontsize=14)
# plt.ylabel('Number of Sources', fontsize=14)
# plt.title('Number Counts Comparison: Observational vs. Synthetic Blackbody Simulation', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()

# # CDF Comparison
# observational_sorted = np.sort(observational_magnitudes)
# synthetic_sorted = np.sort(synthetic_magnitudes_complete)

# observational_cdf = np.arange(1, len(observational_sorted)+1) / len(observational_sorted)
# synthetic_cdf = np.arange(1, len(synthetic_sorted)+1) / len(synthetic_sorted)

# plt.figure(figsize=(10, 6))
# plt.plot(observational_sorted, observational_cdf, label='Observational Data', color='blue')
# plt.plot(synthetic_sorted, synthetic_cdf, label='Synthetic Simulation', color='red', linestyle='--')
# plt.xlabel('Magnitude', fontsize=14)
# plt.ylabel('Cumulative Fraction', fontsize=14)
# plt.title('CDF Comparison: Observational vs. Synthetic Blackbody Simulation', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()

# # Ratio Plot
# ratio = synthetic_counts_complete / (observational_counts)

# plt.figure(figsize=(10, 6))
# plt.plot(mag_bin_centers, ratio, marker='o', linestyle='-', color='purple')
# plt.axhline(1, color='black', linestyle='--')
# plt.xlabel('Magnitude', fontsize=14)
# plt.ylabel('Synthetic / Observational Counts', fontsize=14)
# plt.title('Ratio of Synthetic to Observational Number Counts', fontsize=16)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()

# ------------------------------
# Step 7: Interpretation
# ------------------------------

# Based on the statistical tests and visualizations, interpret the agreement or discrepancies.
# For example:
# if p_value > 0.05 and ks_p_value > 0.05:
#     print("The synthetic simulation is consistent with the observational data.")
# else:
#     print("There are significant differences between the synthetic simulation and the observational data. Investigate potential causes.")

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import h, c, k_B

def blackbody_lambda(wavelength, temperature):
    """
    Calculate the blackbody spectral flux density using Planck's Law.

    Parameters:
    - wavelength (array): Wavelength array in Angstroms.
    - temperature (float): Temperature of the blackbody in Kelvin.

    Returns:
    - flux (array): Spectral flux density in erg/s/cm²/Angstrom.
    """
    wavelength_cm = wavelength * 1e-8  # Convert Angstroms to cm
    exponent = (h.cgs.value * c.cgs.value) / (wavelength_cm * k_B.cgs.value * temperature)
    
    # Prevent overflow by limiting the exponent
    exponent = np.clip(exponent, None, 700)
    
    flux = (2.0 * h.cgs.value * c.cgs.value**2) / (wavelength_cm**5) / (np.exp(exponent) - 1.0)
    return flux  # erg/s/cm²/Angstrom

def gaussian_filter(wave_center, fwhm, amplitude=1.0, num_points=1000):
    """
    Create a Gaussian filter transmission curve.

    Parameters:
    - wave_center (float): Center wavelength in Angstroms.
    - fwhm (float): Full width at half maximum in Angstroms.
    - amplitude (float): Peak transmission.
    - num_points (int): Number of points in the filter curve.

    Returns:
    - wave_filter (array): Wavelength array of the filter.
    - transmission (array): Transmission values.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    wave_min = wave_center - 5 * fwhm
    wave_max = wave_center + 5 * fwhm
    wave_filter = np.linspace(wave_min, wave_max, num_points)
    transmission = amplitude * np.exp(-0.5 * ((wave_filter - wave_center)/sigma)**2) 
    return wave_filter, transmission

def plot_spectrum_and_filter(wavelength, flux, filter_wave, filter_transmission, temperature):
    """
    Plot the blackbody spectrum and the Gaussian filter transmission curve with enhanced aesthetics.

    Parameters:
    - wavelength (array): Wavelength array in Angstroms.
    - flux (array): Spectral flux density in erg/s/cm²/Angstrom.
    - filter_wave (array): Wavelength array of the filter.
    - filter_transmission (array): Transmission values.
    - temperature (float): Temperature of the blackbody in Kelvin.
    """
    plt.style.use('seaborn-darkgrid')  # Use a stylish predefined style
    
    plt.figure(figsize=(12, 8))
    
    # Plot the blackbody spectrum
    plt.plot(wavelength, flux, label=f'Blackbody (T={temperature} K)', color='navy', linewidth=2)
    
    # Scale the filter transmission for visualization purposes
    scaled_transmission = filter_transmission / np.max(filter_transmission) * np.max(flux)
    
    # Plot the Gaussian filter transmission curve with transparency
    plt.plot(filter_wave, scaled_transmission, label='Gaussian Filter', color='crimson', linewidth=2, alpha=0.7)
    
    # Shade the area under the filter transmission curve
    plt.fill_between(filter_wave, 0, scaled_transmission, color='crimson', alpha=0.2)
    
    # Annotations
    # plt.annotate('Peak Blackbody', xy=(wavelength[np.argmax(flux)], np.max(flux)),
    #              xytext=(wavelength[np.argmax(flux)] + 100, np.max(flux) * 0.8),
    #              arrowprops=dict(facecolor='black', shrink=0.05),
    #              fontsize=12, color='navy')
    
    # plt.annotate('Filter Center', xy=(filter_wave[np.argmax(filter_transmission)], scaled_transmission.max()),
    #              xytext=(filter_wave[np.argmax(filter_transmission)] - 200, scaled_transmission.max() * 0.6),
    #              arrowprops=dict(facecolor='black', shrink=0.05),
    #              fontsize=12, color='crimson')
    
    # Customizing the grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Axis labels and title
    plt.xlabel('Wavelength (Angstroms)', fontsize=14, fontweight='bold')
    plt.ylabel('Spectral Flux Density (erg/s/cm²/Angstrom)', fontsize=14, fontweight='bold')
    plt.title('Synthetic Blackbody Spectrum and Gaussian Filter Transmission', fontsize=16, fontweight='bold')
    
    # Legend customization
    plt.legend(fontsize=14, loc='upper right')
    
    # Enhancing spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adding a horizontal line at y=0
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save the figure with high resolution
    plt.savefig('Enhanced_Blackbody_Spectrum_and_Filter.png', dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()

# # Plot blackbody spectrum
#     plt.plot(wavelength.value, flux_photon, label=f'Blackbody (T={temperature} K)', color='blue')

# # Plot Gaussian filter transmission
#     plt.plot(filter_wave, filter_transmission / np.max(filter_transmission),
#          label='Gaussian Filter', color='red')

# # Set axis labels with increased font size
#     plt.xlabel('Wavelength (Angstroms)', fontsize=14)  # Increased from default 12
# p    plt.ylabel('Photon Flux (photons/s/cm²/Angstrom) / Transmission', fontsize=14)

# # Set plot title with increased font size
# plt.title('Synthetic Blackbody Spectrum and Gaussian Filter Transmission', fontsize=16)

# # Adjust tick label sizes
# plt.tick_params(axis='both', which='major', labelsize=12)  # Major ticks
# plt.tick_params(axis='both', which='minor', labelsize=10)  # Minor ticks, if any

# # Set legend with increased font size
# plt.legend(fontsize=12)

# # Enable grid
# plt.grid(True)

# # Optimize layout
# plt.tight_layout()

# # Save the plot with high resolution
# plt.savefig('synthetic_spectrum_high_res.png', dpi=300)

def main():
    # Define parameters for the blackbody
    temperature = 5778  # Kelvin (Sun's temperature)
    wavelength_min = 3000  # Angstroms
    wavelength_max = 10000  # Angstroms
    num_points = 10000  # Number of wavelength points for the spectrum
    
    # Generate the wavelength array
    wavelength = np.linspace(wavelength_min, wavelength_max, num_points)  # Angstroms
    
    # Calculate the blackbody spectral flux density
    flux = blackbody_lambda(wavelength, temperature)
    
    # Define parameters for the Gaussian filter
    filter_center = 6200  # Angstroms
    filter_fwhm = 1200    # Angstroms
    filter_amplitude = 1.0  # Peak transmission
    filter_num_points = 1000  # Number of points in the filter curve
    
    # Create the Gaussian filter transmission curve
    filter_wave, filter_transmission = gaussian_filter(filter_center, filter_fwhm, 
                                                       amplitude=filter_amplitude, 
                                                       num_points=filter_num_points)
    
    # Plot the blackbody spectrum and the filter transmission curve
    plot_spectrum_and_filter(wavelength, flux, filter_wave, filter_transmission, temperature)

if __name__ == "__main__":
    main()


from synphot import SourceSpectrum, Observation, SpectralElement
from astropy import units as u
from synphot import units, SourceSpectrum
from synphot.models import BlackBodyNorm1D, GaussianFlux1D
from astropy.modeling import models as astro_models
import synphot.models as models
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import sys
# Import necessary libraries
from synphot import SourceSpectrum, Observation, SpectralElement
from synphot.models import Empirical1D
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import sys

# Constants for Planck's Law
h = 6.62607015e-27  # Planck constant in erg*s
c = 2.99792458e18   # Speed of light in Angstrom/s
k_B = 1.380649e-16  # Boltzmann constant in erg/K

def planck(wavelength, temperature):
    """
    Calculate the blackbody flux using Planck's Law.

    Parameters:
    - wavelength (array): Wavelength array in Angstroms.
    - temperature (float): Temperature of the blackbody in Kelvin.

    Returns:
    - flux (array): Spectral flux density in erg/s/cm^2/Angstrom.
    """
    wavelength_cm = wavelength * 1e-8  # Convert Angstroms to cm
    exponent = (h * c) / (wavelength_cm * k_B * temperature)
    # Prevent overflow by limiting exponent
    exponent = np.clip(exponent, None, 700)  # Prevent exponent >700
    flux = (2.0 * h * c**2) / (wavelength_cm**5) / (np.exp(exponent) - 1.0)
    return flux  # erg/s/cm^2/Angstrom

def create_blackbody_spectrum(temperature, wavelength_min=5000, wavelength_max=10000, num_points=10000):
    """
    Creates a blackbody spectrum manually using Planck's Law.

    Parameters:
    - temperature (float): Temperature of the blackbody in Kelvin.
    - wavelength_min (float): Minimum wavelength in Angstroms.
    - wavelength_max (float): Maximum wavelength in Angstroms.
    - num_points (int): Number of wavelength points.

    Returns:
    - wavelength (array): Wavelength array in Angstroms.
    - flux_photon (array): Spectral flux density in photons/s/cm^2/Angstrom.
    """
    wavelength = np.linspace(wavelength_min, wavelength_max, num_points) * u.AA  # Angstroms
    flux_energy = planck(wavelength.value, temperature)  # erg/s/cm^2/Angstrom

    # Convert energy flux to photon flux
    wavelength_cm = wavelength.to(u.cm).value  # Convert to cm
    energy_per_photon = (h * c) / wavelength_cm  # erg
    flux_photon = flux_energy / energy_per_photon  # photons/s/cm^2/Angstrom

    # Debugging: Check flux_photon values
    print(f"Flux Photon: min={flux_photon.min()}, max={flux_photon.max()}")

    return wavelength, flux_photon

def create_gaussian_filter(wave_center, fwhm, amplitude=1.0, N=1000):
    """
    Creates a Gaussian filter transmission curve as a SpectralElement.

    Parameters:
    - wave_center (float): Center wavelength in Angstroms.
    - fwhm (float): Full width at half maximum in Angstroms.
    - amplitude (float): Peak transmission.
    - N (int): Number of points in the filter curve.

    Returns:
    - SpectralElement: Gaussian filter transmission curve.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    wave_filter = np.linspace(wave_center - 5*fwhm, wave_center + 5*fwhm, N)  # Angstroms
    transmission = amplitude * np.exp(-0.5 * ((wave_filter - wave_center)/sigma)**2)
    
    # Normalize the transmission curve
    transmission /= np.max(transmission)
    
    # Create Empirical1D model
    transmission_model = Empirical1D(points=wave_filter, lookup_table=transmission)
    
    # Create SpectralElement
    filter_band = SpectralElement(transmission_model)
    
    return filter_band

def create_synphot_spectrum(wavelength, flux_photon):
    """
    Creates a Synphot SourceSpectrum from wavelength and photon flux arrays.

    Parameters:
    - wavelength (array): Wavelength array with units (Angstroms).
    - flux_photon (array): Spectral flux density array in photons/s/cm^2/Angstrom.

    Returns:
    - SourceSpectrum: Synphot SourceSpectrum object.
    """
    try:
        # Create Empirical1D model for the source
        spectrum_model = Empirical1D(points=wavelength.value, lookup_table=flux_photon)
        
        # Create SourceSpectrum
        source = SourceSpectrum(spectrum_model)
        print("Source spectrum created successfully.")
        return source
    except Exception as e:
        print(f"Error creating SourceSpectrum: {e}")
        sys.exit(1)

def perform_photometry(source, filter_band):
    """
    Performs synthetic photometry by convolving the source spectrum with the filter.

    Parameters:
    - source (SourceSpectrum): The source spectrum.
    - filter_band (SpectralElement): The filter bandpass.

    Returns:
    - float: Observed flux in counts/sec.
    """
    try:
        # Create an Observation object
        obs = Observation(source, filter_band, force='extrap')  # Removed 'area' keyword
        print("Observation created successfully.")
        
        # Calculate the observed flux (count rate)
        observed_flux = obs.countrate('ct')  # Use 'ct' as the unit string
        print(f"Observed Flux: {observed_flux:.2f} counts/sec")
        return observed_flux
    except Exception as e:
        print(f"Error during synthetic photometry: {e}")
        sys.exit(1)

def calibrate_magnitude(observed_flux, filter_band):
    """
    Calibrates the magnitude of the source using Vega as the zero-point.

    Parameters:
    - observed_flux (float): Observed flux in counts/sec.
    - filter_band (SpectralElement): The filter bandpass.

    Returns:
    - float: Calibrated magnitude of the source.
    """
    try:
        # Load Vega spectrum
        vega = SourceSpectrum.from_vega()
        print("Vega spectrum loaded successfully.")
        
        # Create an Observation for Vega
        obs_vega = Observation(vega, filter_band, force='extrap')
        vega_flux = obs_vega.countrate('ct')  # Use 'ct' as the unit string
        print(f"Vega Flux: {vega_flux:.2f} counts/sec")
        
        # Define zero-point magnitude (Vega is set to 0)
        zero_point = -2.5 * np.log10(vega_flux)
        print(f"Zero-Point Magnitude: {zero_point:.2f} mag")
        
        # Calculate source magnitude
        magnitude = -2.5 * np.log10(observed_flux) + zero_point
        print(f"Synthetic Source Magnitude: {magnitude:.2f} mag")
        
        return magnitude
    except Exception as e:
        print(f"Error during magnitude calibration: {e}")
        sys.exit(1)

def plot_spectrum_and_filter(wavelength, flux_photon, filter_band, temperature):
    """
    Plots the source spectrum and filter transmission curve.

    Parameters:
    - wavelength (array): Wavelength array in Angstroms.
    - flux_photon (array): Spectral flux density array in photons/s/cm^2/Angstrom.
    - filter_band (SpectralElement): The filter bandpass.
    - temperature (float): Temperature of the blackbody in Kelvin.
    """
    try:
        # Access the filter wavelength array using 'wave' attribute or 'model.points'
        if hasattr(filter_band, 'wave'):
            filter_wave = filter_band.wave  # Correct attribute replacing 'binset'
        else:
            # Access via model.points if 'wave' is not available
            filter_wave = filter_band.model.points
        
        # Access the filter transmission values using 'lookup_table'
        if hasattr(filter_band.model, 'lookup_table'):
            filter_transmission = filter_band.model.lookup_table
        else:
            print("Error: 'lookup_table' attribute not found in the filter's model.")
            sys.exit(1)
        
        # Ensure that filter_wave and filter_transmission are 1D arrays
        filter_wave = np.asarray(filter_wave).flatten()
        filter_transmission = np.asarray(filter_transmission).flatten()
        
        # Verify that both arrays have the same length
        if filter_wave.shape[0] != filter_transmission.shape[0]:
            print(f"Error: Filter wavelength and transmission arrays have different lengths: "
                  f"{filter_wave.shape[0]} vs {filter_transmission.shape[0]}")
            sys.exit(1)
        
        # Debugging: Check flux_photon values
        print(f"Plotting: Flux Photon Range: min={flux_photon.min()}, max={flux_photon.max()}")
        print(f"Plotting: Filter Transmission Range: min={filter_transmission.min()}, max={filter_transmission.max()}")
        
        # Plot the source spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(wavelength.value, flux_photon, label=f'Blackbody (T={temperature} K)', color='blue')
        
        # Plot the filter transmission curve
        plt.plot(filter_wave, filter_transmission / np.max(filter_transmission),
                 label='Gaussian Filter', color='red')
        
        # Labeling the plot
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Photon Flux (photons/s/cm²/Angstrom) / Transmission')
        plt.title('Synthetic Blackbody Spectrum and Gaussian Filter Transmission')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")
        sys.exit(1)

def main():
    # Define blackbody temperature (e.g., Sun's temperature)
    temperature = 5778  # Kelvin
    
    # Step 1: Create Blackbody Spectrum
    wavelength, flux_photon = create_blackbody_spectrum(temperature=temperature,
                                                        wavelength_min=5000,  # Angstroms
                                                        wavelength_max=10000,  # Angstroms
                                                        num_points=10000)
    
    # Step 2: Create a Custom Gaussian Filter
    # Example: Center at 6200 Angstroms with FWHM of 1200 Angstroms
    filter_center = 6200  # Angstroms
    filter_fwhm = 1200    # Angstroms
    filter_band = create_gaussian_filter(wave_center=filter_center,
                                        fwhm=filter_fwhm,
                                        amplitude=1.0,
                                        N=1000)
    print("Custom Gaussian filter created successfully.")
    
    # Step 3: Create Synphot SourceSpectrum from Manual Blackbody
    source = create_synphot_spectrum(wavelength, flux_photon)
    
    # # Step 4: Perform Synthetic Photometry
    # observed_flux = perform_photometry(source, filter_band)
    
    # # Step 5: Calibrate Magnitude with Vega as Zero-Point
    # magnitude = calibrate_magnitude(observed_flux, filter_band)
    
    # Step 6: Plot the Spectrum and Filter Transmission
    plot_spectrum_and_filter(wavelength, flux_photon, filter_band, temperature)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.restoration import denoise_wavelet
from skimage import img_as_float

# ------------------------------
# Step 1: Image Acquisition
# ------------------------------

# Path to your FITS file (replace with your actual file path)
fits_file_path = "/Users/stephi/Desktop/y3-lab/Astro/Fits_Data/mosaic.fits" 

# Read the FITS file
with fits.open(fits_file_path) as hdul:
    image_data = hdul[0].data
    header = hdul[0].header

# Convert image data to float for processing
image_float = img_as_float(image_data)

# ------------------------------
# Step 2: Visualize the Original Image
# ------------------------------

plt.figure(figsize=(12, 6))

# Display the original image using Z-scale for better contrast
plt.subplot(1, 2, 1)
plt.imshow(image_float, cmap='gray', origin='lower')
plt.colorbar()
plt.title('Original Image (Z-scale)')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')

# ------------------------------
# Step 3: Wavelet Denoising
# ------------------------------

# Apply wavelet denoising
# Parameters:
# - sigma: Controls the strength of the denoising. Adjust based on noise level.
# - wavelet: Type of wavelet used. 'db1' (Daubechies wavelet) is common.
# - mode: Signal extension mode. 'soft' is typically used.
# - rescale_sigma: If True, sigma is rescaled by the noise standard deviation.

denoised_image = denoise_wavelet(
    image_float,
    multichannel=False,
    convert2ycbcr=False,
    method='BayesShrink',
    mode='soft',
    wavelet='db1',
    rescale_sigma=True
)

# ------------------------------
# Step 4: Visualize the Denoised Image
# ------------------------------

# Display the denoised image using Z-scale for consistency
plt.subplot(1, 2, 2)
plt.imshow(denoised_image, cmap='gray', origin='lower')
plt.colorbar()
plt.title('Denoised Image (Wavelet)')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')

plt.tight_layout()
plt.show()

# ------------------------------
# Step 5: Compare Histograms
# ------------------------------

# Compute histograms for original and denoised images
bins = 50  # Number of histogram bins

# Original image histogram
counts_orig, bins_orig = np.histogram(image_float, bins=bins, density=True)
bin_centers_orig = (bins_orig[:-1] + bins_orig[1:]) / 2

# Denoised image histogram
counts_denoised, bins_denoised = np.histogram(denoised_image, bins=bins, density=True)
bin_centers_denoised = (bins_denoised[:-1] + bins_denoised[1:]) / 2

# Plot histograms
plt.figure(figsize=(10, 6))
plt.plot(bin_centers_orig, counts_orig, label='Original Image', color='blue')
plt.plot(bin_centers_denoised, counts_denoised, label='Denoised Image', color='red')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability Density')
plt.title('Histogram Comparison: Original vs. Denoised Image')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------
# Step 6: Save the Denoised Image (Optional)
# ------------------------------

# Path to save the denoised FITS file
denoised_fits_path = 'denoised_image.fits'

# Create a new FITS HDU (Header/Data Unit) with the denoised image
hdu_denoised = fits.PrimaryHDU(denoised_image, header=header)

# Write the denoised image to a new FITS file
hdu_denoised.writeto(denoised_fits_path, overwrite=True)

print(f"Denoised image saved to {denoised_fits_path}")


original_image_float = img_as_float(image_data)

# ------------------------------
# Step 2: Wavelet Denoising
# ------------------------------

# Apply wavelet denoising
denoised_image = denoise_wavelet(
    original_image_float,
    multichannel=False,
    convert2ycbcr=False,
    method='BayesShrink',
    mode='soft',
    wavelet='db1',
    rescale_sigma=True
)

# ------------------------------
# Step 3: Define Background Regions
# ------------------------------

def get_background_regions(image, box_size=200):
    """
    Extract background regions from the four corners of the image.

    Parameters:
    - image: 2D numpy array
    - box_size: Size of the square box to extract from each corner

    Returns:
    - List of background pixel values
    """
    top_left = image[:box_size, :box_size].flatten()
    top_right = image[:box_size, -box_size:].flatten()
    bottom_left = image[-box_size:, :box_size].flatten()
    bottom_right = image[-box_size:, -box_size:].flatten()
    return np.concatenate([top_left, top_right, bottom_left, bottom_right])

# Extract background pixels from original and denoised images
background_original = get_background_regions(original_image_float)
background_denoised = get_background_regions(denoised_image)

# ------------------------------
# Step 4: Calculate Standard Deviation
# ------------------------------

# Compute standard deviations
sigma_original = np.std(background_original)
sigma_denoised = np.std(background_denoised)

# Calculate noise reduction percentage
noise_reduction = ((sigma_original - sigma_denoised) / sigma_original) * 100

print(f"Original Background Standard Deviation: {sigma_original:.2f} counts")
print(f"Denoised Background Standard Deviation: {sigma_denoised:.2f} counts")
print(f"Noise Reduction: {noise_reduction:.2f}%")

# ------------------------------
# Step 5: Visual Comparison of Histograms
# ------------------------------

plt.figure(figsize=(10, 6))
plt.hist(background_original, bins=50, alpha=0.5, label='Original Background', color='blue', density=True)
plt.hist(background_denoised, bins=50, alpha=0.5, label='Denoised Background', color='red', density=True)
plt.xlabel('Pixel Counts')
plt.ylabel('Probability Density')
plt.title('Background Pixel Intensity Distribution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------
# Step 6: Visual Comparison of Images (Optional)
# ------------------------------

plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(original_image_float, cmap='gray', origin='lower')
plt.colorbar()
plt.title('Original Image')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')

# Denoised Image
plt.subplot(1, 2, 2)
plt.imshow(denoised_image, cmap='gray', origin='lower')
plt.colorbar()
plt.title('Denoised Image')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')

plt.tight_layout()
plt.show()

