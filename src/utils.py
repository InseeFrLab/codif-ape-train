"""
Utils.
"""
from pathlib import Path


def get_root_path() -> Path:
    """
    Returns root path of project.

    Returns:
        Path: Root path.
    """
    return Path(__file__).parent.parent


# Donnons tord à lino
