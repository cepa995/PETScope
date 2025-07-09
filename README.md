# PETScope

## Package Overview
PETScope is a CLI toolbox for automating end-to-end PET processing pipelines. Currently, PETScope currently utilizes the following two packages:
- [Dynamic PET Analysis](https://github.com/bilgelm/dynamicpet) for dynamic pet modeling (SRTM, Logan Plot, etc.)
- [PETPVC](https://github.com/UCL/PETPVC) for partial volume correction

## Prerequisites
In order to run PETScope CLI, all you need to have installed on your system is:
- `Python 3.11`,
- `docker`, and
- `pip`

## Installation
Steps:
1. Clone repository: `git clone https://github.com/cepa995/PETScope.git`
2. Run: `cd PETScope && pip install -e .`
3. Execute any of the supported End-To-End PET pipelines

**NOTE:** As soon as you run `petscope` CLI for the 1st time, it will automatically pull custom made Docker images which include: **PETScope dependencies** (`PVC`, `DynamicPET`) and **SPM12** (and `MATLAB 2022b runtime`)

## TODO:
1. Interactive end-to-end custom pipeline generation by chaining existing functionalities (e.g., system displays user list of possible inputs, user chooses any number of these inputs, then system based on the choosen inputs spints out to the user which of the functionalities user can chain together once he (the user) provides before mentioned inputs. Once user specifies list of functionalities he wants to chain together into a single pipeline, system will ask for user to provide required inputs and execute the pipeline)
 - Ask if you are performing single subject analysis, or are you analyzing entire study
 - Ask what type of analysis you are doing (e.g. Kinetic Modeling)
 - Build a pipeline (based on the above the settings.json is created for either 1x sub or entire study and a list of available options is being shown to the user)