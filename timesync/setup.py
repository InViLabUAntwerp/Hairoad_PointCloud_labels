import sys
from cx_Freeze import setup, Executable
# python setup.py bdist_msi
import os
# Get the absolute path of the directory containing this setup.py file

# Add this root directory to the system path so cx_Freeze can find the package

# python setup.py bdist_msi
# --- Configuration ---
# Your main application script
main_script = "Example_GUI_folder_to_pcd.py"
# The name of the generated executable
target_name = "Hairoad_lidar_Processor.exe"
# The name of the build output directory
build_dir = "hairoad_processor_build"

# An icon file for your application (must be a .ico file)
icon_file = "processor.ico"
# Your application's name
app_name = "Hairoad Data Processor"
# Your application's version
app_version = "1.0"
# A short description
app_description = "Processes and synchronizes Hairoad master and camera data."

# --- Build Options ---
# Dependencies are automatically detected, but it might need fine-tuning.
# Add modules that cx_Freeze might miss, especially from your custom library.
build_exe_options = {
    # Add packages that your script and its dependencies use.
    # 'hairoad_calib' is your custom package. cx_Freeze needs to know about it.
    # Also include any major libraries it depends on, like 'numpy'.
    "packages": ["os", "wx", "hairoad_calib", "numpy"],

    # List modules to exclude if they are not needed, to reduce build size.
    "excludes": ["tkinter"],

    # If your application needs to include data files (like images, config files),
    # add them here. For example: "include_files": ["my_config.json"].
    "include_files": ["DataClass/","timesync/","timesync/calib1.h5","timesync/transformation_and_intrinsics7.txt", "timesync/Invilab_ico.ico"],
    # Specifies the name of the folder where the built executable will be placed.
    "build_exe": build_dir,
}

# --- Base for GUI Applications ---
# Set the base to "Win32GUI" for a GUI application on Windows.
# This prevents a console window from opening when you run the .exe.
base = None
if sys.platform == "win32":
    base = "Win32GUI"

# --- Setup Configuration ---
setup(
    name=app_name,
    version=app_version,
    description=app_description,
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            main_script,  # Your main script file
            base=base,
            target_name=target_name,
            icon=icon_file  # Sets the file icon for the .exe
        )
    ]
)
