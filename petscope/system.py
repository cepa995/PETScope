import platform
import subprocess
import shutil
import os

from rich import print

def system_check():
    """
    Checks and prints the operating system details and runs the virtualization check.
    """
    # Get OS information
    os_name = platform.system()
    os_version = platform.version()
    os_release = platform.release()

    print(f"[bold green]Operating System:[/] [bold blue]{os_name}")
    print(f"[bold green]OS Version:[/] [bold blue]{os_version}")
    print(f"[bold green]OS Release:[/] [bold blue]{os_release}\n")

    check_virtualization()

def check_virtualization():
    """
    Checks if virtualization is enabled on the current system.
    """
    os_name = platform.system()
    try:
        if os_name == "Windows":
            # Check if 'systeminfo' is available
            if shutil.which('systeminfo') is None:
                print("[bold yellow]Warning: The 'systeminfo' command is not found. Ensure it is available on your system.")
                return
            
            # Check if the script has admin privileges (Windows-specific)
            try:
                output = subprocess.check_output('fltmc', stderr=subprocess.DEVNULL)
                has_admin = True
            except subprocess.CalledProcessError:
                has_admin = False

            if not has_admin:
                print("[bold yellow]Warning: This script may require administrative privileges to run 'systeminfo'.[/]")

            # Run 'systeminfo' to check virtualization
            output = subprocess.check_output(['systeminfo'], text=True)
            if 'Virtualization Enabled In Firmware' in output:
                print("[bold green]Virtualization is enabled.[/]")
            else:
                print("[bold red]Virtualization is not detected.[/]")

        elif os_name == "Linux":
            # Check if 'lscpu' is installed
            if shutil.which('lscpu') is None:
                print("[bold yellow]Warning: The 'lscpu' command is not found. Install it using 'sudo apt install lscpu'.")
                return

            # Check if the user has root permissions (optional but recommended)
            if os.geteuid() != 0:
                print("[bold yellow]Warning: Running as a non-root user. Some system information may be restricted.[/]")

            # Run 'lscpu' to check virtualization
            output = subprocess.check_output(['lscpu'], text=True)
            if 'VT-x' in output or 'AMD-V' in output:
                print("[bold green]Virtualization is enabled.[/]")
            else:
                print("[bold red]Virtualization is not detected.[/]")
        else:
            print("[bold red]This operating system is not supported for virtualization check.[/]")
    except subprocess.CalledProcessError as e:
        print(f"[bold red]Error running system command: {e}[/]")
    except FileNotFoundError:
        print("[bold red]Required command not found. Ensure the necessary utilities are installed.[/]")
    except PermissionError:
        print("[bold red]Permission denied. Try running the script with elevated privileges.[/]")

# Run the system check function
if __name__ == "__main__":
    system_check()
