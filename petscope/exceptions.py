"""Custom Exception Handling"""

# Registration Module Related Exceptions
class ANTsRegistrationException(Exception):
    """Exception raised for errors in the ANTs registration process."""
    pass

class ANTsApplyTransformsException(Exception):
    """Exception raised when applying transforms fails in ANTs."""
    pass

class ANTsUnsupportedTransformType(Exception):
    """Exception raised for unsupported transform types in ANTs."""
    pass

class ANTsImageReadException(Exception):
    """Exception raised when image reading fails in ANTs."""
    pass

# Dynamic PET Utilitis module related Exceptions
class PETJsonGenerationException(Exception):
    """Exception raised when PET JSON file could not be generated due to
    validation related issues"""
    pass

class NotSamePhysicalSpaceException(Exception):
    """Exception raised when two given images are NOT in same space"""
    pass

# PET Utilities Exception
class SavitzkyGolaySmoothingException(Exception):
    """Exception raised when window size is bigger than polynomial 
    order in Savitzky Golay smoothing"""
    pass

class SettingsJSONTemplateNotFoundException(Exception):
    """Exception raised when path to Settings JSON template file
    does not exist"""
    pass

class InvalidSettingsJSONTemplateFileException(Exception):
    """Exception raised when the settings JSON template file
    is invalid (e.g. missing required content, etc.)"""
    pass

class PETImageNotFoundException(Exception):
    """Exception raised if path to the PET image does not exist"""
    pass

class PET3DImageException(Exception):
    """Exception raised if the provided PET image is 3D and not 4D"""
    pass

class SettingsJSONInvalidStructureException(Exception):
    """Exception raised if the settings_template.json file has\
        invalid structure"""
    pass

class FrameNumberMismatchException(Exception):
    """Exception raised if the number of frames in 4D pet image
    does not match number of frames specified in settings_template.json"""
    pass

class FrameStartTimeAndOrDurationException(Exception):
    """Exception raised when the FrameStartTime and FrameDuration lists
    inside settings_template.json are in disagreement"""
    pass