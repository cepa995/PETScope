from typer.testing import CliRunner
from petscope import __app_name__, __version__, cli

# Create a CLI runner 
runner = CliRunner()

def test_version():
    """Testing Application's Version"""
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} v{__version__}\n" in result.stdout