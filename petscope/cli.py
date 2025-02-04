"""This module provides the PETScope CLI"""

import typer
import numpy as np
from rich import print
from typing import Optional
from typing_extensions import Annotated
from petscope import __app_name__, __version__
from petscope.petscope import PETScope
from petscope.system import system_check
from petscope.utils import read_settings_json


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

@app.command(name="compute_tac")
def get_tac(
    pet_image_path: Annotated[str, typer.Argument(help="Absolute path to the PET 4D Image")],
    template_path: Annotated[str, typer.Argument(help="Absolute path to the Template Image")],
    time_activity_curve_out: Annotated[str, typer.Argument(help="Absolute path to TAC output (.png) image")],
    window_size: int = typer.Option(None, "--window_size", "-w", help="Choose Window Size for TAC Smoothing", rich_help_panel="Window Size for Time Activity Curve Smoothing"),
    polynomial_order: int = typer.Option(None, "--polyorder", "-p", help="Choose Polynomial Order for Savitzky Golay TAC smoothing", rich_help_panel="Polynomial Order for Savitzky Golay TAC smoothing"),
    reference_name: str = typer.Option(None, "--reference", "-r", help="Choose one of the Supported Reference Region (WholeCerebellum, WhiteMatter)", rich_help_panel="Supported Reference Regions"),
    template_name: str = typer.Option(None, "--template", "-t", help="Choose name of the Template which was passed under template_path argument (FreeSurfer)", rich_help_panel="Supported Templates"),
) -> np.array:
    """
    Computes Time Activity Curve (TAC).

    Calculates the TAC over the specified reference region using the provided PET 4D image,
    template, and optional smoothing parameters.

    Args:
        pet_image_path (str): Absolute path to the PET 4D image.
        template_path (str): Absolute path to the template image.
        time_activity_curve_out (str): Absolute path to save the TAC plot (.png).
        window_size (int, optional): Window size for TAC smoothing. Defaults to None.
        polynomial_order (int, optional): Polynomial order for TAC smoothing. Defaults to None.
        reference_name (str, optional): Name of the reference region. Defaults to None.
        template_name (str, optional): Name of the template. Defaults to None.

    Returns:
        np.array: Computed TAC.
    """
    print(f":gear: [bold green]Computing Time Activity Curve (TAC) over the {reference_name}")
    petscope = get_petscope()
    error_code = petscope.get_tac(
        pet_image_path=pet_image_path,
        template_path=template_path,
        template_name=template_name,
        reference_name=reference_name,
        time_activity_curve_out=time_activity_curve_out,
        window_length=window_size,
        polyorder=polynomial_order
    )
    if error_code:
        print(f":x: [bold red]Could NOT Compute Time Activity Curve over the {reference_name}! ")
    else:
        print(f":white_heavy_check_mark: [bold green]Successfully Computed TAC Over the {reference_name}! ")


@app.command(name="pet_to_t1")
def coregister_pet_and_mr(
    pet_4d_path: Annotated[str, typer.Argument(help="Absolute path to the PET 4D Image")],
    t1_3d_path: Annotated[str, typer.Argument(help="Absolute path to the T1 3D Image")],
    output_dir: Annotated[str, typer.Argument(help="Absolute path to the directory (does not have to exist), where result will be stored")],
    type_of_transform: str = typer.Option("Rigid", "--transform", "-t", help="Choose Transformation Type (Rigid, Affine, SyN)", rich_help_panel="Transformation Types"),
) -> None:
    """
    Registers a 4D PET image to T1 space.

    Computes a mean 3D volume from the provided 4D PET image and performs image registration
    using ANTs to align it with the T1 image.

    Args:
        pet_4d_path (str): Absolute path to the PET 4D image.
        t1_3d_path (str): Absolute path to the T1 3D image.
        output_dir (str): Directory to store the registration results.
        type_of_transform (str): Transformation type (Rigid, Affine, SyN). Defaults to "Rigid".
    """
    petscope = get_petscope()
    print(f"\n:fire: [bold yellow]Starting PET -> T1 ANTs {type_of_transform} Registration! :fire:")
    error_code = petscope.coregister_pet_and_mr(
        pet_4d_path=pet_4d_path,
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
        target_region: str = typer.Option("Hippocampus", "--target", "-tar", help="Reference region (Hippocampus)."),
        model: str = typer.Option("SRTMZhou2003", "--model", "-m", help="SRTM model to use."),
        pvc_method: str = typer.Option(None, "--pvc_method", "-pvc", help="Partial Volume Correction method."),
        window_size: int = typer.Option(None, "--window_size", "-w", help="Window size for TAC smoothing."),
        polynomial_order: int = typer.Option(None, "--polyorder", "-p", help="Polynomial order for TAC smoothing."),
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
        target_region (str): Target region to use. Defaults to "Hippocampus".
        model (str): SRTM model to use. Defaults to "SRTMZhou2003".
        pvc_method (str, optional): Partial Volume Correction method. Defaults to None.
        window_size (int, optional): Window size for TAC smoothing. Defaults to None.
        polynomial_order (int, optional): Polynomial order for TAC smoothing. Defaults to None.
    """
    # Perform a system check before running the pipeline
    system_check()

    # Load PET settings from the JSON configuration
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
        pet_json=pet_json
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