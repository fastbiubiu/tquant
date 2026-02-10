"""
Pytest configuration for tquant tests.
Configures Python path to include tquant directory.
"""

import sys
from pathlib import Path

# Add tquant directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "tquant"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
