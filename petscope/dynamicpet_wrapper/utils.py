import os
import json
from petscope.exceptions import PETJsonGenerationException
from petscope.constants import REQUIRED_KEYS, PET_DATA

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_settings_template():
    """Load the settings template JSON file."""
    settings_path = os.path.join(ROOT_DIR, 'settings_template.json')
    with open(settings_path, 'r') as file:
        return json.load(file)

SETTINGS_JSON_DATA = load_settings_template()

def generate_subject_json(subject_pet_4d_path):
    """
    Generate a PET JSON file for a given subject.

    :param subject_pet_4d_path: Absolute path to a 4D PET image,
        which is located in a directory where the subject PET JSON
        should be created.
    """
    file_path, _ = os.path.splitext(subject_pet_4d_path)
    subject_pet_json = f"{file_path}.json"
    
    pet_data = SETTINGS_JSON_DATA.get(PET_DATA)
    if pet_data is None:
        raise PETJsonGenerationException(f"{PET_DATA} section is missing in settings.")

    # Validate and write JSON data
    try:
        _is_json_valid(pet_data)
        with open(subject_pet_json, 'w') as f:
            json.dump(pet_data, f)
    except (KeyError, ValueError) as e:
        raise PETJsonGenerationException(f"PET JSON data is not valid: {e}")

def _is_json_valid(json_object):
    """
    Validates that the JSON object has required keys with non-empty values.

    :param json_object: The JSON data to validate.
    :return: True if valid, raises exception otherwise.
    """
    missing_keys = REQUIRED_KEYS - json_object.keys()
    if missing_keys:
        raise KeyError(f"Missing keys in JSON: {', '.join(missing_keys)}")
    
    empty_values = [
        key for key in REQUIRED_KEYS
        if json_object.get(key) in ("", [], {}, None)
    ]
    if empty_values:
        raise ValueError(f"Keys with empty values: {', '.join(empty_values)}")
    
    return True

if __name__ == "__main__":
    subject_pet_4d_path = "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/data/pet_3d.nii"
    generate_subject_json(subject_pet_4d_path)
