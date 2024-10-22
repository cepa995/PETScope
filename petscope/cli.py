"""This module provides the PETScope CLI"""

import typer
from rich import print
from typing import Optional, List
from typing_extensions import Annotated
from petscope import __app_name__, __version__
from petscope.petscope import PETScope

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

@app.command(name="run_srtm")
def run_srtm(
        pet_4d_path: Annotated[str, typer.Argument(help="Absolute path to the PET 4D Image")],
        t1_3d_path: Annotated[str, typer.Argument(help="Absolute path to the T1 3D Image")],
        template_path: Annotated[str, typer.Argument(help="Absolute path to the Template (e.g. FreeSurfer) Mask in T1 Space")],
        output_dir: Annotated[str, typer.Argument(help="Absolute path to the directory (does not have to exist), where result will be stored")],
        template: str = typer.Option("FreeSurfer", "--tmpl", "-t", help="Choose a template (FreeSurfer)", rich_help_panel="Templates"),
        reference_region: str = typer.Option("WholeCerebellum", "--ref", "-r", help="Choose a reference region/mask (WholeCerebellum, WhiteMatter)", rich_help_panel="Reference Region"),
        model: str = typer.Option("SRTMZhou2003", "--model", "-m", help="Choose SRTM Model (SRTMZhou2003)", rich_help_panel="Available SRTM Model")
) -> None:
    """Runs SRTM Pipeline"""
    petscope = get_petscope()
    print("\n:fire: [bold yellow]Starting Simplified Tissue Model (SRTM) Pipeline! :fire:")
    error_code = petscope.run_srtm(
        pet_4d_path=pet_4d_path,
        t1_3d_path=t1_3d_path,
        template=template,
        template_path=template_path,
        reference_region=reference_region,
        output_dir=output_dir,
        model=model
    )
    if error_code:
        print(":x: [bold red]SRTM Pipeline Was NOT Successful! ")
    else:
        print(":white_heavy_check_mark: [bold green]SRTM Pipeline Ran Successfully! ")