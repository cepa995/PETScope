import platform
import subprocess
import shutil
import os

from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from petscope.constants import SPM_DOCKER_IMAGE, PET_DEP_DOCKER_IMAGE

def check_and_pull_docker_images():
    render_docker_logo()
    docker_images = [SPM_DOCKER_IMAGE, PET_DEP_DOCKER_IMAGE]
    for image in docker_images:
        print(f"[bold blue]:information:[bold blue][bold yellow] Checking for " + 
              f"[blue]{image}[/] Docker Image")
        try:
            subprocess.check_call(["docker", "pull", image])
            print(f":white_check_mark:[bold green] Successfully pulled {image}\n")
        except subprocess.CalledProcessError as e:
            print(f"[bold red]Failed to pull Docker image {image}: {e}\n")

def render_docker_logo():
    # Create a representation of the Docker logo using text
    logo = r"""
                     ##         .
               ## ## ##        ==
            ## ## ## ## ##    ===
        /=================\___/ ===
   ~~~ {~~ ~~~~ ~~~ ~~~~ ~~~ ~ /  ===- ~~~
        \______ o          __/
         \    \        __/
          \____\______/
    """
# Simulated docker image names for demonstration purposes
    docker_images = ["SPM_DOCKER_IMAGE", "PET_DEP_DOCKER_IMAGE"]

    # Create a console to capture and format output
    console = Console(record=True)
    
    # Initialize Text object to accumulate formatted output
    output_text = Text()
    output_text.append(logo + "\n\n", style="bold blue")

    for image in docker_images:
        # Format the checking message
        checking_message = Text(f"ℹ️  Checking for {image} Docker Image\n", style="bold yellow")
        output_text.append(checking_message)

        try:
            # Simulate successful docker pull output (replace with actual call as needed)
            # subprocess.check_call(["docker", "pull", image])
            success_message = Text(f"✔️  Successfully pulled {image}\n\n", style="bold green")
            output_text.append(success_message)
        except subprocess.CalledProcessError as e:
            error_message = Text(f"❌ Failed to pull Docker image {image}: {e}\n\n", style="bold red")
            output_text.append(error_message)

    # Create a panel that includes the Docker logo and the captured formatted output
    panel = Panel(output_text, border_style="blue", title="Docker Whale with Pull Output")

    # Print the panel to the console
    console.print(panel)

def system_check():
    """
    Checks and prints the operating system details and runs the virtualization check.
    """
    # Get OS information
    os_name = platform.system()
    os_version = platform.version()
    os_release = platform.release()
    is_enabled = check_virtualization()

    table = Table(title="System Information")

    table.add_column("Operating System", justify="center", style="cyan", no_wrap=True)
    table.add_column("Operating System Version", justify="center", style="cyan", no_wrap=True)
    table.add_column("Operating System Release", justify="center", style="cyan", no_wrap=True)
    table.add_column("Virtualization Enabled", justify="center", style="cyan", no_wrap=True)
 
    table.add_row(os_name, os_version, os_release, str(is_enabled))

    console = Console()
    console.print(table)
    console.print('\n')
    render_docker_logo()

def check_virtualization():
    """
    Checks if virtualization is enabled on the current system.
    """
    os_name = platform.system()
    try:
        if os_name == "Windows":
            # Check if 'systeminfo' is available
            if shutil.which('systeminfo') is None:
                print("[bold yellow]Warning: The 'systeminfo' command is not found. Ensure it is available on your system.\n")
                return False
            
            # Check if the script has admin privileges (Windows-specific)
            try:
                output = subprocess.check_output('fltmc', stderr=subprocess.DEVNULL)
                has_admin = True
            except subprocess.CalledProcessError:
                has_admin = False

            if not has_admin:
                print("[bold yellow]Warning: This script may require administrative privileges to run 'systeminfo'.[/]\n")

            # Run 'systeminfo' to check virtualization
            output = subprocess.check_output(['systeminfo'], text=True)
            if 'Virtualization Enabled In Firmware' in output:
                print("[bold green]Virtualization is enabled.[/]")
                return True
            else:
                print("[bold red]Virtualization is not detected. Will not be able to perform visual QC :sad_but_relieved_face:[/]\n")
                return False

        elif os_name == "Linux":
            # Check if 'lscpu' is installed
            if shutil.which('lscpu') is None:
                print("[bold yellow]Warning: The 'lscpu' command is not found. Install it using 'sudo apt install lscpu'.\n")
                return False

            # Check if the user has root permissions (optional but recommended)
            if os.geteuid() != 0:
                print("[bold yellow]Warning: Running as a non-root user. Some system information may be restricted.[/]\n")

            # Run 'lscpu' to check virtualization
            output = subprocess.check_output(['lscpu'], text=True)
            if 'VT-x' in output or 'AMD-V' in output:
                print("[bold green]Virtualization is enabled.[/]")
                return True
            else:
                print("[bold red]Virtualization is not detected.  Will not be able to perform visual QC :sad_but_relieved_face:[/]\n")
                return False
        else:
            print("[bold red]This operating system is not supported for virtualization check. :sad_but_relieved_face:[/]")
    except subprocess.CalledProcessError as e:
        print(f"[bold red]Error running system command: {e}[/]")
    except FileNotFoundError:
        print("[bold red]Required command not found. Ensure the necessary utilities are installed.[/]")
    except PermissionError:
        print("[bold red]Permission denied. Try running the script with elevated privileges.[/]")

# Run the system check function
if __name__ == "__main__":
    system_check()
