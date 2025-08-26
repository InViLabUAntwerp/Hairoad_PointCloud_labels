import sys
from cx_Freeze import setup, Executable
# python setup_pcd_label.py bdist_msi
# Dependencies are automatically detected, but it might need fine-tuning.
# Add modules that might be missed here.
build_exe_options = {
    "packages": ["os", "numpy", "cv2","vispy","open3d"],
    "excludes": ["tkinter"],
    # Tell cx_Freeze to copy your DataClass folder and the icon
    "include_files": [
        "DataClass/",
        "genpycam.ico",
        "Logo_InViLab_2023_Horizontal_color.png"
    ],
"build_exe": "pcd_player_label_build",  # Specify the build directory
}

# Set the base for a GUI application on Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="PCD Player",
    version="1.0",
    description="A player for .pcd pointcloud files",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "pcd_player_label.py",          # Your main script file
            base=base,
            target_name="PCD Label Player.exe",
            icon="genpycam.ico"       # This sets the file icon for the .exe
        )
    ]
)