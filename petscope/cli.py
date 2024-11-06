"""This module provides the PETScope CLI"""

import typer
import numpy as np
from rich import print
from typing import Optional, List
from typing_extensions import Annotated
from petscope import __app_name__, __version__
from petscope.petscope import PETScope
from petscope.utils import read_settings_json

# Create explicit typer application
app = typer.Typer()

def _version_callback(value: bool) -> None:
    """
    Version Callback
    
    :param value - whether or not to print application's
     name and version using echo()
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
    return

def get_petscope() -> PETScope:
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
    """Computes Time Activity Curve (TAC) over the specified
    reference region"""
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
def pet_to_t1(
    pet_4d_path: Annotated[str, typer.Argument(help="Absolute path to the PET 4D Image")],
    t1_3d_path: Annotated[str, typer.Argument(help="Absolute path to the T1 3D Image")],
    output_dir: Annotated[str, typer.Argument(help="Absolute path to the directory (does not have to exist), where result will be stored")],
    type_of_transform: str = typer.Option("Rigid", "--transform", "-t", help="Choose Transformation Type (Rigid, Affine, SyN)", rich_help_panel="Transformation Types"),
) -> None:
    """Computes a mean 3D volume from a given 4D PET image and registers it using ANTs
    to T1 space"""
    petscope = get_petscope()
    print(f"\n:fire: [bold yellow]Starting PET -> T1 ANTs {type_of_transform} Registration! :fire:")
    error_code = petscope.pet_to_t1(
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
        template_path: Annotated[str, typer.Argument(help="Absolute path to the Template (e.g. FreeSurfer) Mask in T1 Space")],
        output_dir: Annotated[str, typer.Argument(help="Absolute path to the directory (does not have to exist), where result will be stored")],
        template: str = typer.Option("FreeSurfer", "--tmpl", "-t", help="Choose a template (FreeSurfer)", rich_help_panel="Templates"),
        reference_region: str = typer.Option("WholeCerebellum", "--ref", "-r", help="Choose a reference region/mask (WholeCerebellum, WhiteMatter)", rich_help_panel="Reference Region"),
        model: str = typer.Option("SRTMZhou2003", "--model", "-m", help="Choose SRTM Model (SRTMZhou2003)", rich_help_panel="Available SRTM Model"),
        window_size: int = typer.Option(None, "--window_size", "-w", help="Choose Window Size for TAC Smoothing", rich_help_panel="Window Size for Time Activity Curve Smoothing"),
        polynomial_order: int = typer.Option(None, "--polyorder", "-p", help="Choose Polynomial Order for Savitzky Golay TAC smoothing", rich_help_panel="Polynomial Order for Savitzky Golay TAC smoothing")
) -> None:
    """Runs SRTM Pipeline"""
    # Load PET JSON file
    pet_json = read_settings_json(pet_4d_path)
    # Get PETScope object and execute SRTM Pipeline
    petscope = get_petscope()
    print("\n:fire: [bold yellow]Starting Simplified Tissue Model (SRTM) Pipeline! :fire:")
    error_code = petscope.run_srtm(
        pet_4d_path=pet_4d_path,
        t1_3d_path=t1_3d_path,
        template=template,
        template_path=template_path,
        reference_region=reference_region,
        output_dir=output_dir,
        model=model,
        window_size=window_size,
        polynomial_order=polynomial_order,
        pet_json=pet_json
    )
    if error_code:
        print(":x: [bold red]SRTM Pipeline Was NOT Successful! ")
    else:
        print(":white_heavy_check_mark: [bold green]SRTM Pipeline Ran Successfully! ")