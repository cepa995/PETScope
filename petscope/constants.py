import os

# Supported CLI Tasks
# -------------------------
# Defines list of supported CLI tasks and their corresponding file extensions
SUPPORTED_TASKS_DICT = {
    "registration" : [".nii", ".nii.gz"]
}

# Supported Kinetic Pet Modeling models
# ---------------------------
# Defines list of models for which wrapper function has been implemented
SUPPORTED_DYNAMICPET_MODELS = {
    "SRTMZhou2003"
}

# Suported k2prime estimation methods
# ---------------------------
# Defines list of k2prime estimation methods for 1st SRTM pass
SUPPORTED_K2PRIME_METHODS = [
    "voxel_based",
    "tac_based"
]

# Custom Pipeline constants
# -------------------------
# Defines dictionary for supported commands which can be used to create custom pipeline
# and a dictionary which maps these commands to specific functions within the toolbox
CUSTOM_PIPELINE_COMMAND_DICT= {
    "Coregister PET 4D to MRI (and vice-versa)": "Adding Coregistration of the PET native to MRI (and vice versa) space to the pipeline",
    "PET Specific Commands": {
        "SPM Realignment": "Adding SPM Realignment step of the PET native image to the pipeline",
        "Partial Volume Correction (PVC)": "Adding Partial Volume Correction (PVC) step to the pipeline",
        "Compute Time Activity Curve": "Adding Time Activity Curve (TAC) computation to the pipeline",
    },
    "MRI Mask Template Specific Commands": {
        "Compute Reference Mask": "Adding Reference Mask computation to the pipeline",
    },
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

# Reference Regions Used for Dynamic PET modeling
# ---------------------------------------
# Lists the supported reference regions used for Dynamic PET modeling
SUPPORTED_REFERENCE_REGIONS = ['WholeCerebellum', 'WholeWhiteMatter']

# Target Regions Used for Dynamic PET Modelig (e.g. Logan Plot)
# ---------------------------------------
# Lists the supported reference regions used for Dynamic PET modeling
SUPPORTED_TARGET_REGIONS = ['Hippocampus']

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

# Target Regions for Dynamic PET Modeling
# ------------------------------------------
# TARGET_REGIONS: Defines list of target regions supported for dynamic PET modeling. These regions are grouped by
# template (e.g., FreeSurfer) and further subdivided into specific reference regions.
# First, create FREESURFER_LABELS by inverting the FREESURFER_REGIONS dictionary
FREESURFER_LABELS = {v: k for k, v in FREESURFER_REGIONS.items()}

# Now complete the TARGET_REGIONS dictionary
TARGET_REGIONS = {
    "FreeSurfer": {
        "Hippocampus": [
            int(FREESURFER_LABELS["Left-Hippocampus"]),
            int(FREESURFER_LABELS["Right-Hippocampus"]),
        ],
        "Frontal cortex": [
            # Left hemisphere frontal regions
            int(FREESURFER_LABELS["ctx-lh-caudalmiddlefrontal"]),
            int(FREESURFER_LABELS["ctx-lh-parsopercularis"]),
            int(FREESURFER_LABELS["ctx-lh-parsorbitalis"]),
            int(FREESURFER_LABELS["ctx-lh-parstriangularis"]),
            int(FREESURFER_LABELS["ctx-lh-rostralmiddlefrontal"]),
            int(FREESURFER_LABELS["ctx-lh-superiorfrontal"]),
            int(FREESURFER_LABELS["ctx-lh-frontalpole"]),
            # Right hemisphere frontal regions
            int(FREESURFER_LABELS["ctx-rh-caudalmiddlefrontal"]),
            int(FREESURFER_LABELS["ctx-rh-parsopercularis"]),
            int(FREESURFER_LABELS["ctx-rh-parsorbitalis"]),
            int(FREESURFER_LABELS["ctx-rh-parstriangularis"]),
            int(FREESURFER_LABELS["ctx-rh-rostralmiddlefrontal"]),
            int(FREESURFER_LABELS["ctx-rh-superiorfrontal"]),
            int(FREESURFER_LABELS["ctx-rh-frontalpole"]),
        ],
        "Inferior temporal cortex": [
            int(FREESURFER_LABELS["ctx-lh-inferiortemporal"]),
            int(FREESURFER_LABELS["ctx-rh-inferiortemporal"]),
        ],
        "Cerebellar cortex": [
            int(FREESURFER_LABELS["Left-Cerebellum-Cortex"]),
            int(FREESURFER_LABELS["Right-Cerebellum-Cortex"]),
        ],
        "Centrum semiovale": [
            # This is primarily deep white matter regions
            int(FREESURFER_LABELS["Left-Cerebral-White-Matter"]),
            int(FREESURFER_LABELS["Right-Cerebral-White-Matter"]),
        ],
        "Occipital cortex": [
            int(FREESURFER_LABELS["ctx-lh-lateraloccipital"]),
            int(FREESURFER_LABELS["ctx-lh-cuneus"]),
            int(FREESURFER_LABELS["ctx-lh-pericalcarine"]),
            int(FREESURFER_LABELS["ctx-rh-lateraloccipital"]),
            int(FREESURFER_LABELS["ctx-rh-cuneus"]),
            int(FREESURFER_LABELS["ctx-rh-pericalcarine"]),
        ],
        "Parietal cortex": [
            int(FREESURFER_LABELS["ctx-lh-inferiorparietal"]),
            int(FREESURFER_LABELS["ctx-lh-superiorparietal"]),
            int(FREESURFER_LABELS["ctx-lh-supramarginal"]),
            int(FREESURFER_LABELS["ctx-lh-postcentral"]),
            int(FREESURFER_LABELS["ctx-lh-precuneus"]),
            int(FREESURFER_LABELS["ctx-rh-inferiorparietal"]),
            int(FREESURFER_LABELS["ctx-rh-superiorparietal"]),
            int(FREESURFER_LABELS["ctx-rh-supramarginal"]),
            int(FREESURFER_LABELS["ctx-rh-postcentral"]),
            int(FREESURFER_LABELS["ctx-rh-precuneus"]),
        ],
        "Cingulate gyrus": [
            int(FREESURFER_LABELS["ctx-lh-caudalanteriorcingulate"]),
            int(FREESURFER_LABELS["ctx-lh-isthmuscingulate"]),
            int(FREESURFER_LABELS["ctx-lh-posteriorcingulate"]),
            int(FREESURFER_LABELS["ctx-lh-rostralanteriorcingulate"]),
            int(FREESURFER_LABELS["ctx-rh-caudalanteriorcingulate"]),
            int(FREESURFER_LABELS["ctx-rh-isthmuscingulate"]),
            int(FREESURFER_LABELS["ctx-rh-posteriorcingulate"]),
            int(FREESURFER_LABELS["ctx-rh-rostralanteriorcingulate"]),
        ],
        "Putamen": [
            int(FREESURFER_LABELS["Left-Putamen"]),
            int(FREESURFER_LABELS["Right-Putamen"]),
        ],
        "Thalamus": [
            int(FREESURFER_LABELS["Left-Thalamus-Proper"]),
            int(FREESURFER_LABELS["Right-Thalamus-Proper"]),
        ],
        "Entorhinal cortex": [
            int(FREESURFER_LABELS["ctx-lh-entorhinal"]),
            int(FREESURFER_LABELS["ctx-rh-entorhinal"]),
        ],
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
        "CerebralWhite": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "-Cerebral-White-Matter" in label
        ],
        "CorpusCallosum": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "CC_" in label
        ],
        "InferiorCerebellarGray": [
            int(key) for key, label in FREESURFER_REGIONS.items() if "-Cerebellum-Cortex" in label or "-Cerebellum-White-Matter" in label
        ],
    }
}