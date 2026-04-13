# conftest.py
# Makes the wind_model package importable when running pytest from any directory
# without needing sys.path manipulation in individual test files.

import sys
from pathlib import Path

# Add the wind_model root (one level above tests/) to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
