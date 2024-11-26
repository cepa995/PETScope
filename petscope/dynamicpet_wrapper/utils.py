import os
import json
from petscope.exceptions import PETJsonGenerationException
from petscope.constants import REQUIRED_KEYS, PET_DATA

# Define the root directory for settings template
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_settings_template() -> dict:
    """
    Load the settings template JSON file.

    This function reads the `settings_template.json` file located in the project root directory 
    and returns its content as a dictionary.

    Returns:
        dict: Parsed JSON data from the settings template.

    Raises:
        FileNotFoundError: If the `settings_template.json` file does not exist.
        json.JSONDecodeError: If the JSON file has an invalid structure.
    """
    settings_path = os.path.join(ROOT_DIR, 'settings_template.json')
    with open(settings_path, 'r') as file:
        return json.load(file)

# Load the settings JSON data into a constant
SETTINGS_JSON_DATA = load_settings_template()

def generate_subject_json(subject_pet_4d_path: str) -> str:
    """
    Generate a PET JSON file for a given subject based on the settings template.

    This function validates and writes a JSON file for a specific subject, ensuring 
    that all required keys and values are present and valid.

    Args:
        subject_pet_4d_path (str): Absolute path to a 4D PET image. 
            The JSON file will be created in the same directory with the same base name.

    Returns:
        str: Path to the generated subject JSON file.

    Raises:
        PETJsonGenerationException: If the `PET_DATA` section is missing in the settings 
            or if the JSON data is invalid.
        KeyError: If required keys are missing in the JSON object.
        ValueError: If any required keys have empty or invalid values.

    Example:
        generate_subject_json("/path/to/pet_4d_image.nii")
    """
    # Generate the file path for the output JSON
    file_path, _ = os.path.splitext(subject_pet_4d_path)
    subject_pet_json = f"{file_path}.json"
    
    # Retrieve the PET data section from the settings
    pet_data = SETTINGS_JSON_DATA.get(PET_DATA)
    if pet_data is None:
        raise PETJsonGenerationException(f"{PET_DATA} section is missing in settings.")

    # Validate and write the JSON data to a file
    try:
        _is_json_valid(pet_data)
        with open(subject_pet_json, 'w') as f:
            json.dump(pet_data, f, indent=4)
    except (KeyError, ValueError) as e:
        raise PETJsonGenerationException(f"PET JSON data is not valid: {e}")

    return subject_pet_json

def _is_json_valid(json_object: dict) -> bool:
    """
    Validates the JSON object to ensure it contains all required keys with non-empty values.

    Args:
        json_object (dict): The JSON data to validate.

    Returns:
        bool: True if the JSON object is valid.

    Raises:
        KeyError: If required keys are missing in the JSON object.
        ValueError: If any required keys have empty or invalid values.

    Example:
        _is_json_valid({"key1": "value", "key2": 42})
    """
    # Check for missing keys
    missing_keys = set(REQUIRED_KEYS) - set(json_object.keys())
    if missing_keys:
        raise KeyError(f"Missing keys in JSON: {', '.join(missing_keys)}")
    
    # Check for empty values
    empty_values = [
        key for key in REQUIRED_KEYS
        if json_object.get(key) in ("", [], {}, None)
    ]
    if empty_values:
        raise ValueError(f"Keys with empty values: {', '.join(empty_values)}")
    
    return True
