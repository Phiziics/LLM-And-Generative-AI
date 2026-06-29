# Import sys so we can add the project root to Python path
import sys

# Import Path for reliable file path handling
from pathlib import Path

# Get the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))