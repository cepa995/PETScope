# PETScope

## Package Overview
PETScope is a CLI toolbox for automating end-to-end PET processing pipelines. Currently, PETScope relies heavly on the following two packages:
- [Dynamic PET Analysis](https://github.com/bilgelm/dynamicpet) for dynamic pet modeling (SRTM, Logan Plot, etc.)
- [PETPVC](https://github.com/UCL/PETPVC) for partial volume correction

## Install
### `pip`
Prerequisites:
- `DynamicPET`
- `PETPVC`

Steps:
1. Clone and install `DynamicPET` package by following instructions from [here](https://github.com/bilgelm/dynamicpet/tree/main)
2. Either build and install `PETPVC v1.2.4`, or download executable binaries from [here](https://github.com/UCL/PETPVC/releases/download/v1.2.4/PETPVC-1.2.4-Linux.tar.gz) and store them in your `/opt/` directory
3. Clone PETScope repository: `git clone https://github.com/cepa995/PETScope.git`
4. Install PETScope package: `pip install -e PETScope`

### `docker`
If you do not have `DynamicPET` and `PETPVC` setup on your machine, you can install `PETScope` with `docker`! Here are the steps that you need to follow:
1. Clone PETScope repository: `git clone https://github.com/cepa995/PETScope.git`
2. Change to `build/` directory where `Dockerfile` is stored: `cd PETScope && cd build`
3. Build `docker` image: `docker build -t <image-name>:<tag> .`
4. Run `docker` image: `docker run <image-name>:<tag> --help` 

## Running PETScope from Docker
Prerequisites:
- `docker`

Steps:
1. Pull an existing `docker` image: `docker pull stefancepa995/petscope:latest`
2. Run `docker` image: `docker run stefancepa995/petscope:latest --help`
