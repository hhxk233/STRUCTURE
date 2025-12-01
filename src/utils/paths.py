import os
from pathlib import Path

current_directory = os.path.dirname(os.path.abspath(__file__))
project_path = Path(
    os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
)
data_path = project_path / "data"
configs_path = project_path / "configs"
