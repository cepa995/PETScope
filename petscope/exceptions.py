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