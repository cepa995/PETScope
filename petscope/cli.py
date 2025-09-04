"""This module provides the PETScope CLI"""

import typer
import numpy as np
from rich import print
from typing import Optional
from typing_extensions import Annotated
from petscope import __app_name__, __version__
from petscope.petscope import PETScope
from petscope.system import system_check
from petscope.utils import read_settings_json, is_ext_supported


# Initialize Typer Application
# -----------------------------
# The main Typer application is defined, which will serve as the entry point
# for all CLI commands provided by PETScope.
app = typer.Typer()

def _version_callback(value: bool) -> None:
    """
    Version Callback

    Prints the application's name and version when the --version/-v option is provided.

    Args:
        value (bool): Whether to print the version and exit.

    Raises:
        typer.Exit: Exits the CLI after printing the version information.
    """
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show PETScope's version and exit",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    """
    PETScope CLI Main Entry Point.

    Provides a version option for displaying the application version.
    """
    return

def get_petscope() -> PETScope:
    """
    Returns an instance of the PETScope class.

    This function is a helper to ensure consistent initialization
    of the PETScope object across commands.
    """
    return PETScope()

@app.command(name="pet_to_mri")
def coregister_pet_and_mr(
    pet_path: Annotated[str, typer.Argument(help="Absolute path to the PET 3D or 4D Image")],
    t1_3d_path: Annotated[str, typer.Argument(help="Absolute path to the T1 3D Image")],
    output_dir: Annotated[str, typer.Argument(help="Absolute path to the directory (does not have to exist), where result will be stored")],
    type_of_transform: str = typer.Option("Rigid", "--transform", "-t", help="Choose Transformation Type (Rigid, Affine, SyN)", rich_help_panel="Transformation Types"),
) -> None:
    """
    Registers a 4D PET image to T1 space.

    Computes a mean 3D volume from the provided 4D PET image and performs image registration
    using ANTs to align it with the T1 image.

    Args:
        pet_path (str): Absolute path to the PET 3D or 4D image.
        t1_3d_path (str): Absolute path to the T1 3D image.
        output_dir (str): Directory to store the registration results.
        type_of_transform (str): Transformation type (Rigid, Affine, SyN). Defaults to "Rigid".
    """
    # Check if the extensions are supported for a given task
    if is_ext_supported(pet_path, 'registration'):
        print(f"\n:x: Could not execute coregistration due to the invalid file extension for the PET image")
    elif is_ext_supported(t1_3d_path, 'registration'):
        print(f"\n:x: Could not execute coregistration due to the invalid file extension for the MR image")

    # Get PETScope object
    petscope = get_petscope()

    # Start coregistration of the PET image to T1 image
    print(f"\n:fire: [bold yellow]Starting Coregistration using ANTs {type_of_transform}! :fire:")
    error_code = petscope.coregister_pet_and_mr(
        pet_path=pet_path,
        t1_3d_path=t1_3d_path,
        type_of_transform=type_of_transform,
        output_dir=output_dir
    )
    if error_code:
        print(":x: [bold red]PET to T1 Registration Was NOT Successful! ")
    else:
        print(":white_heavy_check_mark: [bold green]PET t T1 Registration was Successfull! ")

@app.command(name="run_srtm")
def run_srtm(
        pet_4d_path: Annotated[str, typer.Argument(help="Absolute path to the PET 4D Image")],
        t1_3d_path: Annotated[str, typer.Argument(help="Absolute path to the T1 3D Image")],
        template_path: Annotated[str, typer.Argument(help="Absolute path to the Template Mask in T1 Space")],
        output_dir: Annotated[str, typer.Argument(help="Directory to store the SRTM results.")],
        template: str = typer.Option("FreeSurfer", "--tmpl", "-t", help="Template to use (e.g., FreeSurfer)."),
        physical_space: str = typer.Option("MRI", "--space", "-s", help="Space for computation (MRI or PET)."),
        reference_region: str = typer.Option("WholeCerebellum", "--ref", "-r", help="Reference region (WholeCerebellum, WholeWhiteMatter)."),
        target_region: str = typer.Option(None, "--target", "-tar", help="Target region"),
        model: str = typer.Option(None, "--model", "-m", help="SRTM model to use (SRTMZhou2003)."),
        pvc_method: str = typer.Option(None, "--pvc_method", "-pvc", help="Partial Volume Correction method."),
        window_size: int = typer.Option(None, "--window_size", "-w", help="Window size for TAC smoothing."),
        polynomial_order: int = typer.Option(None, "--polyorder", "-p", help="Polynomial order for TAC smoothing."),
        k2prime_estimation_method: str = typer.Option("tac_based", "--k2prime_estimation_method", "-k2p", help="Method for k2 prime estimation during 1st SRTM pass ('voxel_based', 'tac_based')."),
) -> None:
    """
    Runs the Simplified Reference Tissue Model (SRTM) Pipeline.

    This command performs SRTM analysis on a 4D PET image using specified templates,
    reference regions, and models. Partial Volume Correction (PVC) is optional.

    Args:
        pet_4d_path (str): Absolute path to the PET 4D image.
        t1_3d_path (str): Absolute path to the T1 3D image.
        template_path (str): Absolute path to the template mask in T1 space.
        output_dir (str): Directory to store the SRTM results.
        template (str): Template to use (e.g., FreeSurfer). Defaults to "FreeSurfer".
        physical_space (str): Space for computation (MRI or PET). Defaults to "MRI".
        reference_region (str): Reference region to use. Defaults to "WholeCerebellum".
        target_region (str): Target region to use. By default the command will compute stats for all available target regions.
        model (str): *IMPORTANT* This argument indicates use of https://github.com/bilgelm/dynamicpet. SRTM model to use. Defaults to "SRTMZhou2003".
        pvc_method (str, optional): Partial Volume Correction method. Defaults to None.
        window_size (int, optional): Window size for TAC smoothing. Defaults to None.
        polynomial_order (int, optional): Polynomial order for TAC smoothing. Defaults to None.
        k2prime_estimation_method (str,optional): *IMPORTANT* This argument is for CUSTOM SRTM2 implementation only! Can be voxel_based or tac_based
    """
    # Perform a system check before running the pipeline
    system_check()

    # Load PET settings from the JSON configuration and Validate
    pet_json = read_settings_json(pet_4d_path)

    # Initialize PETScope and execute the SRTM pipeline
    petscope = get_petscope()
    print("\n:fire: [bold yellow]Starting Simplified Tissue Model (SRTM) Pipeline! :fire:")
    error_code = petscope.run_srtm(
        pet_4d_path=pet_4d_path,
        t1_3d_path=t1_3d_path,
        template=template,
        template_path=template_path,
        physical_space=physical_space,
        reference_region=reference_region,
        target_region=target_region,
        output_dir=output_dir,
        model=model,
        pvc_method=pvc_method,
        window_size=window_size,
        polynomial_order=polynomial_order,
        pet_json=pet_json,
        k2prime_estimation_method=k2prime_estimation_method
    )
    if error_code:
        print(":x: [bold red]SRTM Pipeline Was NOT Successful! ")
    else:
        print(":white_heavy_check_mark: [bold green]SRTM Pipeline Ran Successfully! ")


@app.command(name="run_custom_pipeline")
def run_custom_pipeline() -> None:
    # Initialize PETScope and execute the SRTM pipeline
    petscope = get_petscope()
    petscope.custom_pipeline()