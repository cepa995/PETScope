"""
Custom Exception Handling

This module defines custom exceptions used throughout the PETScope framework. 
Each exception is tailored to handle specific errors encountered during image 
registration, PET utilities, dynamic PET modeling, and other related tasks.

Exceptions are grouped by the modules or functionalities they pertain to.
"""

# Registration Module Related Exceptions
class ANTsRegistrationException(Exception):
    """
    Exception raised for errors encountered during the ANTs registration process.

    This exception is triggered when the ANTs registration fails due to 
    invalid parameters, corrupted input data, or runtime errors.
    """
    pass

class ANTsApplyTransformsException(Exception):
    """
    Exception raised when applying transformations fails in ANTs.

    This exception is raised when the application of forward or inverse 
    transformations encounters an error during the ANTs pipeline.
    """
    pass

class ANTsUnsupportedTransformType(Exception):
    """
    Exception raised for unsupported transform types in ANTs.

    This exception is raised when the specified transformation type 
    is not included in the list of supported ANTs transformations.
    """
    pass

class ANTsImageReadException(Exception):
    """
    Exception raised when image reading fails in ANTs.

    This exception is triggered when ANTs cannot read an input image, 
    possibly due to unsupported formats or invalid file paths.
    """
    pass

class PhysicalSpaceSupportException(Exception):
    """
    Exception raised when the specified physical space is unsupported.

    This exception is triggered when the user provides a physical space 
    that is not supported by the current system configuration.
    """
    pass

# Dynamic PET Utilities Module Related Exceptions
class PETJsonGenerationException(Exception):
    """
    Exception raised when PET JSON file generation fails.

    This exception is triggered when validation-related issues prevent 
    the creation of a valid PET JSON configuration file.
    """
    pass

class NotSamePhysicalSpaceException(Exception):
    """
    Exception raised when two images are not in the same physical space.

    This exception is triggered when spatial inconsistencies between 
    the provided images are detected during validation.
    """
    pass

class SRTMDynamicPETException(Exception):
    """
    Exception raised when Dynamic PET SRTM computation fails.

    This exception is triggered when the Simplified Reference Tissue 
    Model (SRTM) fails to process due to invalid inputs or configuration.
    """
    pass

# PET Utilities Exceptions
class SavitzkyGolaySmoothingException(Exception):
    """
    Exception raised for invalid Savitzky-Golay smoothing parameters.

    This exception is triggered when the window size is smaller than 
    the polynomial order or other invalid configurations are detected.
    """
    pass

class SettingsJSONTemplateNotFoundException(Exception):
    """
    Exception raised when the Settings JSON template file is not found.

    This exception is triggered when the specified path to the Settings 
    JSON template file does not exist or is inaccessible.
    """
    pass

class InvalidSettingsJSONTemplateFileException(Exception):
    """
    Exception raised for invalid Settings JSON template files.

    This exception is triggered when the Settings JSON template file 
    is missing required content, has an invalid structure, or contains 
    malformed JSON data.
    """
    pass

class PETImageNotFoundException(Exception):
    """
    Exception raised when the specified PET image file is not found.

    This exception is triggered when the file path to the PET image does 
    not exist or is inaccessible.
    """
    pass

class PET3DImageException(Exception):
    """
    Exception raised when the provided PET image is 3D instead of 4D.

    This exception is triggered when a 4D PET image is required for 
    processing but the provided image is 3D.
    """
    pass

class SettingsJSONInvalidStructureException(Exception):
    """
    Exception raised for invalid structure in the Settings JSON template.

    This exception is triggered when the Settings JSON file has an 
    incorrect structure or is missing required keys or values.
    """
    pass

class FrameNumberMismatchException(Exception):
    """
    Exception raised when the number of PET frames does not match the JSON configuration.

    This exception is triggered when the number of frames in the 4D PET image 
    differs from the number specified in the Settings JSON template.
    """
    pass

class FrameStartTimeAndOrDurationException(Exception):
    """
    Exception raised for inconsistencies in frame start times or durations.

    This exception is triggered when the FrameStartTime and FrameDuration 
    lists in the Settings JSON file are misaligned or inconsistent.
    """
    pass

class PETDataUnitsException(Exception):
    """
    Exception raised when PET data is not in the required units (kBq/mL).

    This exception is triggered when the voxel intensities of the PET data 
    are not normalized to kBq/mL, which is required for further processing.
    """
    pass

# SPM Wrapper Related Exception
class SPMRealignmentException(Exception):
    """
    Exception raised during SPM realignment when an error occurs.

    This exception is triggered when the SPM realignment process fails due 
    to invalid inputs, missing dependencies, or runtime errors.
    """
    pass

# PVC Related Exceptions
class PVCMethodSupportException(Exception):
    """
    Exception raised for unsupported PVC methods.

    This exception is triggered when the user specifies a Partial Volume 
    Correction (PVC) method that is not supported by the system.
    """
    pass
