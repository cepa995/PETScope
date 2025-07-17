# PETScope

A comprehensive CLI toolbox for automating end-to-end PET (Positron Emission Tomography) processing pipelines. PETScope streamlines complex PET analysis workflows by providing an intuitive command-line interface that orchestrates multiple specialized tools and packages.

## ⚠️ Platform Support

**Currently, PETScope only supports Linux platforms.** Support for Windows and macOS is planned for future releases.

## Overview

PETScope is designed to simplify and automate PET image analysis by providing a unified interface for common PET processing tasks. The toolbox integrates several established packages and tools to create seamless, reproducible analysis pipelines.

### Current Capabilities

- **Dynamic PET Modeling**: SRTM (Simplified Reference Tissue Model), Logan Plot analysis, and other kinetic modeling approaches
- **Partial Volume Correction**: Advanced correction techniques for improved quantification accuracy
- **Docker-based Execution**: Containerized environment ensures reproducibility and eliminates dependency conflicts
- **End-to-End Automation**: Complete analysis pipelines from raw data to final results

## Dependencies and Integrations

PETScope currently utilizes the following packages:

- **[Dynamic PET Analysis](https://github.com/bilgelm/dynamicpet)**: For dynamic PET modeling (SRTM, Logan Plot, etc.)
- **[PETPVC](https://github.com/UCL/PETPVC)**: For partial volume correction
- **SPM12**: Statistical Parametric Mapping with MATLAB 2022b runtime (automatically provided via Docker)

## System Requirements

- **Operating System**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.11 or higher
- **Docker**: Latest stable version
- **pip**: Python package installer
- **Storage**: At least 10GB free space (for Docker images and temporary files)

## Installation

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cepa995/PETScope.git
   cd PETScope
   ```

2. **Install PETScope:**
   ```bash
   pip install -e .
   ```

3. **First Run Setup:**
   ```bash
   petscope --help
   ```

**Note**: On first execution, PETScope will automatically pull custom Docker images containing:
- PETScope dependencies (PVC, DynamicPET)
- SPM12 with MATLAB 2022b runtime

### Verification

To verify your installation:
```bash
petscope --version
```

## Usage

### Basic Command Structure

```bash
petscope [COMMAND] [OPTIONS] [ARGUMENTS]
```

### Available Commands

- `petscope run_srtm`: Runs an end-to-end SRTM pipeline
- `petscope pet_to_mri`: Coregistration between PET and MR data

### Image Processing
- **Partial Volume Correction**: Multiple algorithms available
- **Motion Correction**: Frame-to-frame alignment
- **Spatial Normalization**: Standard space registration
- **Smoothing and Filtering**: Customizable parameters

## Planned Features

### Interactive Pipeline Builder
- Guided workflow creation through CLI prompts
- Dynamic input validation and suggestion
- Custom pipeline templates
- Export/import pipeline configurations

### Enhanced Analysis Options
- Single subject vs. study-wide analysis selection
- Statistical analysis integration
- Report generation with visualizations

### Platform Expansion
- Windows support (planned)
- macOS support (planned)

## Data Format Requirements

PETScope accepts the following input formats:
- **NIfTI** (.nii, .nii.gz): Primary format for PET images

## Troubleshooting

### Common Issues

**Docker Permission Errors:**
```bash
sudo usermod -aG docker $USER
# Log out and back in, or restart your session
```

**Memory Issues:**
- Ensure sufficient RAM (8GB minimum)
- Consider processing smaller ROIs or downsampling for large datasets

**Path Issues:**
- Use absolute paths for input/output directories
- Ensure proper file permissions for output directories

### Getting Help

- Check the [Issues](https://github.com/cepa995/PETScope/issues) page for known problems
- Use `petscope --help` for command-specific help
- Enable verbose output with `--verbose` flag for debugging

## Acknowledgments

PETScope builds upon several excellent open-source projects:
- Dynamic PET Analysis by Murat Bilgel
- PETPVC by the UCL team
- SPM12 by the Wellcome Centre for Human Neuroimaging

## Contact

- **Author**: Stefan Milorad Radonjić (cepa995)
- **Repository**: https://github.com/cepa995/PETScope
- **Issues**: https://github.com/cepa995/PETScope/issues

---

**Version**: 1.0.0-beta  