import argparse

def init_argparse() -> argparse.ArgumentParser:
    """ Initialize command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_results_dir", "-trd", required=True, help="Path to test results directory"
    )
    return parser