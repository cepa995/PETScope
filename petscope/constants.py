import os

# Custom Pipeline constants
# -------------------------
# Defines dictionary for supported commands which can be used to create custom pipeline
# and a dictionary which maps these commands to specific functions within the toolbox
CUSTOM_PIPELINE_COMMAND_DICT= {
    "Coregister PET to MRI (and vice-versa)": "Adding Coregistration of the PET native to MRI (and vice versa) space to the pipeline",
    "PET Specific Commands": {
        "SPM Realignment": "Adding SPM Realignment step of the PET native image to the pipeline",
        "Partial Volume Correction (PVC)": "Adding Partial Volume Correction (PVC) step to the pipeline",
        "Compute PET Mean Image": "Adding Mean image computation to the pipeline",
        "Compute Time Activity Curve": "Adding Time Activity Curve (TAC) computation to the pipeline",
    },
    "MRI Mask Template Specific Commands": {
        "Compute Reference Mask": "Adding Reference Mask computation to the pipeline",
    },
}
CUSTOM_PIPELINE_FUNCTION_DICT = {

}

# Supported Physical Spaces
# -------------------------
# Defines the physical spaces supported by the framework for image registration and analysis.
SUPPORTED_PHYSICAL_SPACES = ["MRI", "PET"]
MRI_PHYSICAL_SPACE = "MRI"
PET_PHYSICAL_SPACE = "PET"

# Docker Images
# -------------
# Defines the Docker images required for running the PETScope CLI, including dependencies for
# SPM and other PETScope tools.
SPM_DOCKER_IMAGE = 'stefancepa995/matlab-spm-2022b_runtime:latest'
PET_DEP_DOCKER_IMAGE = 'stefancepa995/petscope-dependencies:latest'

# Partial Volume Correction (PVC) Methods
# ---------------------------------------
# Lists the supported methods for Partial Volume Correction (PVC) used during PET image processing.
PVC_SUPPORTED_METHODS = ['IterativeYang']

# Reference Regions Used for PVC
# ---------------------------------------
# Lists the supported reference regions used for Partial Volume Correction (PVC)
SUPPORTED_REFERENCE_REGIONS = ['WholeCerebellum', 'WholeWhiteMatter']

# Directory Paths
# ----------------
# ROOT_DIR: Defines the root directory for the PETScope project.
# SETTINGS_JSON: Absolute path to the settings JSON template file used for configuring PET processing pipelines.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTINGS_JSON = os.path.join(ROOT_DIR, 'settings_template.json')

# ANTs Registration Transformations
# ---------------------------------
# Lists the types of transformations supported by the ANTs image registration framework.
SUPPORTED_ANTS_TRANSFORM_TYPES = ['Rigid', 'Affine']

# JSON Template Keys
# -------------------
# PET_DATA: Parent key for PET-related settings in the JSON configuration template.
# REQUIRED_KEYS: Lists all the mandatory keys in the JSON template for PET data processing.
PET_DATA = 'pet_json'
REQUIRED_KEYS =  [
    "AcquisitionMode", "AttenuationCorrection", "BodyPart", "FrameDuration", 
    "FrameTimesStart", "ImageDecayCorrected", "ImageDecayCorrectionTime", 
    "InjectedMass", "InjectedMassUnits", "InjectedRadioactivity", 
    "InjectedRadioactivityUnits", "InjectionEnd", "InjectionStart", 
    "Manufacturer", "ManufacturersModelName", "ModeOfAdministration", 
    "ReconFilterSize", "ReconFilterType", "ReconMethodName", 
    "ReconMethodParameterLabels", "ReconMethodParameterUnits", 
    "ReconMethodParameterValues", "ScanStart", "SpecificRadioactivity", 
    "SpecificRadioactivityUnits", "TimeZero", "TracerName", 
    "TracerRadionuclide", "Units"
]

# FreeSurfer Regions and Thresholds
# ---------------------------------
# FS_REGION_THRESHOLDS: Defines threshold values for specific FreeSurfer regions used during image analysis.
# FREESURFER_REGIONS: Maps FreeSurfer labels to human-readable region names for both left and right hemispheres.
FS_REGION_THRESHOLDS = {
    2: 0.911416817,
    3: 1.062319498,
    4: 0.724545528,
    5: 0.915279104,
    41: 0.917129024,
    42: 1.038031116,
    43: 0.729486856,
    44: 0.895404525,
    8: 1.011509206,
    47: 1.020246469,
    14: 0.757979424,
    15: 0.824953957,
    16: 0.749197819,
    24: 0.990087905,
    28: 0.869447067,
    60: 0.859131459,
    30: 1.042621657,
    62: 1.045713417,
    31: 0.907818397,
    63: 0.904097931,
    72: 0.761247946,
    77: 0.813496142,
    80: 1.029712578,
    85: 0.896382004,
    251: 0.893359572,
    252: 0.841278305,
    253: 0.7967451,
    254: 0.793692922,
    255: 0.814210531,
    1000: 0.856684903,
    2000: 0.875955493,
    1006: 1.090389025,
    2006: 1.141628372,
    17: 0.885475281,
    53: 0.892317365,
    1016: 1.01004686117474,
    2016: 0.990927277,
    1007: 1.084620762,
    2007: 1.077052406,
    1013: 1.084154342,
    2013: 1.078481598,
    18: 0.813014127,
    54: 0.841580013,
    1015: 1.163940211,
    2015: 1.149319559,
    10: 0.81265515,
    49: 0.801225245,
    1002: 0.895643751,
    2002: 0.904890982,
    1026: 0.881135238,
    2026: 0.876418485,
    1023: 0.925190109,
    2023: 0.924255045,
    1010: 0.945483627,
    2010: 0.934382846,
    1035: 0.915929094,
    2035: 0.922217962,
    1009: 1.165458947,
    2009: 1.152743882,
    1033: 1.123122683,
    2033: 1.142862953,
    1028: 0.97732317,
    2028: 0.988382384,
    1012: 1.078060797,
    2012: 1.071917838,
    1014: 0.923237819,
    2014: 0.934152604,
    1032: 1.22654434,
    2032: 1.244929092,
    1003: 1.036013896,
    2003: 1.040084892,
    1027: 1.103491084,
    2027: 1.115100755,
    1018: 1.021749661,
    2018: 1.018778742,
    1019: 1.302141531,
    2019: 1.244777912,
    1020: 1.120092782,
    2020: 1.117848646,
    11: 0.752748422,
    50: 0.760585287,
    12: 0.908839222,
    51: 0.906769586,
    1011: 1.229452488,
    2011: 1.218238882,
    1031: 1.085296913,
    2031: 1.052749815,
    1008: 1.161238687,
    2008: 1.140519643,
    1030: 1.085497597,
    2030: 1.083870785,
    13: 0.896384465,
    52: 0.883138437,
    1029: 1.050175254,
    2029: 1.05239517,
    1025: 0.98555858,
    2025: 0.983792005,
    1001: 1.058326664,
    2001: 1.010447057,
    26: 0.811819852,
    58: 0.814165541,
    1034: 0.967990481,
    2034: 0.96326876,
    1021: 1.049285845,
    2021: 1.049951321,
    1022: 1.030848525,
    2022: 1.023259115,
    1005: 1.122145749,
    2005: 1.124075593,
    1024: 0.983139961,
    2024: 0.984922744,
    1017: 0.94413741,
    2017: 0.940599092,
}

FREESURFER_REGIONS = {
    # General and unknown regions
    "0": "Unknown",
    "1": "Cerebellar-Crus",

    # Left hemisphere regions
    "2": "Left-Cerebral-White-Matter",
    "3": "Left-Cerebral-Cortex",
    "4": "Left-Lateral-Ventricle",
    "5": "Left-Inf-Lat-Vent",
    "7": "Left-Cerebellum-White-Matter",
    "8": "Left-Cerebellum-Cortex",
    "10": "Left-Thalamus-Proper",
    "11": "Left-Caudate",
    "12": "Left-Putamen",
    "13": "Left-Pallidum",
    "14": "Third-Ventricle",
    "15": "Fourth-Ventricle",
    "16": "Brain-Stem",
    "17": "Left-Hippocampus",
    "18": "Left-Amygdala",
    "24": "CSF",
    "26": "Left-Accumbens-area",
    "28": "Left-VentralDC",
    "29": "Left-undetermined",
    "30": "Left-vessel",
    "31": "Left-choroid-plexus",

    # Right hemisphere regions
    "41": "Right-Cerebral-White-Matter",
    "42": "Right-Cerebral-Cortex",
    "43": "Right-Lateral-Ventricle",
    "44": "Right-Inf-Lat-Vent",
    "46": "Right-Cerebellum-White-Matter",
    "47": "Right-Cerebellum-Cortex",
    "49": "Right-Thalamus-Proper",
    "50": "Right-Caudate",
    "51": "Right-Putamen",
    "52": "Right-Pallidum",
    "53": "Right-Hippocampus",
    "54": "Right-Amygdala",
    "58": "Right-Accumbens-area",
    "60": "Right-VentralDC",
    "62": "Right-vessel",
    "63": "Right-choroid-plexus",

    # Other regions
    "72": "5th-Ventricle",
    "77": "WM-hypointensities",
    "80": "non-WM-hypointensities",
    "85": "Optic-Chiasm",
    "173": "Midbrain",
    "174": "Pons",
    "175": "Medulla",
    "178": "SCP",
    "251": "CC_Posterior",
    "252": "CC_Mid_Posterior",
    "253": "CC_Central",
    "254": "CC_Mid_Anterior",
    "255": "CC_Anterior",

    # Left cortex regions
    "1000": "ctx-lh-unknown",
    "1001": "ctx-lh-bankssts",
    "1002": "ctx-lh-caudalanteriorcingulate",
    "1003": "ctx-lh-caudalmiddlefrontal",
    "1005": "ctx-lh-cuneus",
    "1006": "ctx-lh-entorhinal",
    "1007": "ctx-lh-fusiform",
    "1008": "ctx-lh-inferiorparietal",
    "1009": "ctx-lh-inferiortemporal",
    "1010": "ctx-lh-isthmuscingulate",
    "1011": "ctx-lh-lateraloccipital",
    "1012": "ctx-lh-lateralorbitofrontal",
    "1013": "ctx-lh-lingual",
    "1014": "ctx-lh-medialorbitofrontal",
    "1015": "ctx-lh-middletemporal",
    "1016": "ctx-lh-parahippocampal",
    "1017": "ctx-lh-paracentral",
    "1018": "ctx-lh-parsopercularis",
    "1019": "ctx-lh-parsorbitalis",
    "1020": "ctx-lh-parstriangularis",
    "1021": "ctx-lh-pericalcarine",
    "1022": "ctx-lh-postcentral",
    "1023": "ctx-lh-posteriorcingulate",
    "1024": "ctx-lh-precentral",
    "1025": "ctx-lh-precuneus",
    "1026": "ctx-lh-rostralanteriorcingulate",
    "1027": "ctx-lh-rostralmiddlefrontal",
    "1028": "ctx-lh-superiorfrontal",
    "1029": "ctx-lh-superiorparietal",
    "1030": "ctx-lh-superiortemporal",
    "1031": "ctx-lh-supramarginal",
    "1032": "ctx-lh-frontalpole",
    "1033": "ctx-lh-temporalpole",
    "1034": "ctx-lh-transversetemporal",
    "1035": "ctx-lh-insula",

    # Right cortex regions
    "2000": "ctx-rh-unknown",
    "2001": "ctx-rh-bankssts",
    "2002": "ctx-rh-caudalanteriorcingulate",
    "2003": "ctx-rh-caudalmiddlefrontal",
    "2005": "ctx-rh-cuneus",
    "2006": "ctx-rh-entorhinal",
    "2007": "ctx-rh-fusiform",
    "2008": "ctx-rh-inferiorparietal",
    "2009": "ctx-rh-inferiortemporal",
    "2010": "ctx-rh-isthmuscingulate",
    "2011": "ctx-rh-lateraloccipital",
    "2012": "ctx-rh-lateralorbitofrontal",
    "2013": "ctx-rh-lingual",
    "2014": "ctx-rh-medialorbitofrontal",
    "2015": "ctx-rh-middletemporal",
    "2016": "ctx-rh-parahippocampal",
    "2017": "ctx-rh-paracentral",
    "2018": "ctx-rh-parsopercularis",
    "2019": "ctx-rh-parsorbitalis",
    "2020": "ctx-rh-parstriangularis",
    "2021": "ctx-rh-pericalcarine",
    "2022": "ctx-rh-postcentral",
    "2023": "ctx-rh-posteriorcingulate",
    "2024": "ctx-rh-precentral",
    "2025": "ctx-rh-precuneus",
    "2026": "ctx-rh-rostralanteriorcingulate",
    "2027": "ctx-rh-rostralmiddlefrontal",
    "2028": "ctx-rh-superiorfrontal",
    "2029": "ctx-rh-superiorparietal",
    "2030": "ctx-rh-superiortemporal",
    "2031": "ctx-rh-supramarginal",
    "2032": "ctx-rh-frontalpole",
    "2033": "ctx-rh-temporalpole",
    "2034": "ctx-rh-transversetemporal",
    "2035": "ctx-rh-insula",

    # Additional regions
    "60001": "Cerebellar-Crus",
}

# Reverse mapping of FreeSurfer region labels for quick lookups.
FREESURFER_LABELS = {label: key for key, label in FREESURFER_REGIONS.items()}

# Reference Regions for Dynamic PET Modeling
# ------------------------------------------
# REFERENCE_REGIONS: Defines reference regions supported for dynamic PET modeling. These regions are grouped by
# template (e.g., FreeSurfer) and further subdivided into specific reference regions.
REFERENCE_REGIONS = {
    "FreeSurfer": {
        "BrainStem": [
            int(FREESURFER_LABELS["Brain-Stem"]),
            int(FREESURFER_LABELS["Pons"]),
            int(FREESURFER_LABELS["Medulla"]),
            int(FREESURFER_LABELS["Midbrain"]),
            int(FREESURFER_LABELS["SCP"]),
        ],
        "CerebellarGray": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "-Cerebellum-Cortex" in label
        ],
        "CerebellarWhite": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "-Cerebellum-White-Matter" in label
        ],
        "WholeCerebellum": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "-Cerebellum-Cortex" in label or "-Cerebellum-White-Matter" in label
        ],
        "CerebralWhite": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "-Cerebral-White-Matter" in label
        ],
        "WholeWhiteMatter": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "-Cerebral-White-Matter" in label or "-Cerebellum-White-Matter" in label
        ],
        "CorpusCallosum": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "CC_" in label
        ],
        "InferiorCerebellarGray": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "-Cerebellum-Cortex" in label or "-Cerebellum-White-Matter" in label
        ],
    }
}