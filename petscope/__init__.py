"""Top-Level Package for PETScope.

This module contains the application name, version, and error code definitions 
used throughout the PETScope application.

Attributes:
    __app_name__ (str): The name of the PETScope application.
    __version__ (str): The current version of the PETScope application.
    SUCCESS (int): Error code representing a successful operation.
    DIR_ERROR (int): Error code for configuration directory-related issues.
    FILE_ERROR (int): Error code for configuration file-related issues.
    JSON_ERROR (int): Error code for invalid JSON file handling.
    ID_ERROR (int): Error code for invalid to-do item IDs.
    ERRORS (dict): A dictionary mapping error codes to descriptive error messages.
"""

# Application metadata
__app_name__ = "petscope"
__version__ = "0.1.0"

# Error Codes
(
    SUCCESS,      # Operation completed successfully
    DIR_ERROR,    # Configuration directory error
    FILE_ERROR,   # Configuration file error
    JSON_ERROR,   # JSON-related error
    ID_ERROR,     # To-do ID error (used for a to-do feature, if applicable)
) = range(5)

# Error Messages
ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    ID_ERROR: "to-do id error",
}
