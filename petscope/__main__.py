"""PETScope Entry Point Script.

This script initializes the PETScope Command Line Interface (CLI) application.
It invokes the main `typer` app defined in the `petscope.cli` module and sets
the program name to the application's name (`__app_name__`).

To run the PETScope CLI, execute this script directly.
"""

from petscope import cli, __app_name__

def main():
    """
    Entry point for the PETScope CLI.

    Invokes the `typer` application defined in the `cli` module, setting the
    program name for better CLI user experience.

    Example:
        $ python -m petscope
        (or)
        $ petscope
    """
    cli.app(prog_name=__app_name__)

if __name__ == "__main__":
    # If executed directly, call the main function to start the CLI.
    main()
