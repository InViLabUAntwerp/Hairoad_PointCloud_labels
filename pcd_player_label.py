#
# copyright Seppe Sels 2025
#
# This code is for internal use only (Uantwerpen, project members)
# Bugs, bugfixes and additions to the code need to be reported to Invilab (contact: Seppe Sels)
# GUI: AI generated
# This script provides a GUI for visualizing PCD/PLY point cloud files,
# including features for normal visualization, custom coloring, and profile analysis.


import wx
import numpy as np
import os
import sys
import threading
import open3d as o3d
import json
import csv

# Optional import for LOWESS smoothing for profile fitting.

from statsmodels.nonparametric.smoothers_lowess import lowess

statsmodels_available = True

# Imports for 3D visualization using VisPy.
from vispy import scene
from vispy.scene import cameras
from vispy.scene.visuals import Text, Line
from vispy.color import Color, get_colormap

# Add PIL import for image loading
from PIL import Image

# Imports for Matplotlib for histogram and profile plotting.
# Note: This adds a new dependency. If building with cx_Freeze,
# 'matplotlib' should be added to the packages list in setup.py.
import matplotlib

matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller. """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def open_point_cloud_file(filepath):
    """
    Opens a .pcd or .ply file using open3d and returns the point and color data.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None
    try:
        # o3d.io.read_point_cloud can handle both .pcd and .ply files
        pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.has_points():
            print(f"Warning: File loaded but contains no points: {filepath}")
            return None, None

        points = np.asarray(pcd.points)
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)

        print(f"Opened {os.path.basename(filepath)} with {len(points)} points.")
        return points, colors
    except Exception as e:
        print(f"Failed to open point cloud file: {filepath}\nError: {e}")
        return None, None


class PCDViewerFrame(wx.Frame):
    def __init__(self, *args, filepath=None, **kw):
        super(PCDViewerFrame, self).__init__(*args, **kw)

        self.points = None
        self.colors = None
        self.normals = None
        self.show_normals = False
        self.show_colors = False
        self.flip_z = False
        self.color_by_z = True  # Enable Z-coloring by default
        self.default_point_color = 'white'

        # Y-clipping attributes
        self.clip_by_y = False  # Disable Y-clipping by default for consistency
        self.y_data_min = 0.0
        self.y_data_max = 1.0
        self.y_clip_min = 0.0
        self.y_clip_max = 1.0
        self.y_min_slider = None
        self.y_max_slider = None

        # Index-clipping attributes
        self.clip_by_index = False
        self.index_data_min = 0
        self.index_data_max = 1
        self.index_clip_min = 0
        self.index_clip_max = 1
        self.index_min_slider = None
        self.index_max_slider = None

        self.z_data_min = 0.0
        self.z_data_max = 1.0
        self.z_color_min = 0.0
        self.z_color_max = 1.0
        self.z_min_slider = None
        self.z_max_slider = None

        # Z-Scale attributes
        self.z_scale_factor = 1.0
        self.z_scale_slider = None

        self.current_dir = None
        self.file_list = []
        self.current_file_index = -1

        self.canvas = None
        self.view = None
        self.markers = None
        self.normals_visual = None
        self.toggle_colors_item = None
        self.flip_z_item = None
        self.color_by_z_item = None
        self.clip_by_y_item = None
        self.clip_by_index_item = None
        self.calculate_normals_item = None
        self.show_normals_item = None
        self.dark_bg_item = None

        self.hist_canvas = None
        self.hist_figure = None
        self.hist_axes = None
        self.hist_min_line = None
        self.hist_max_line = None

        # XZ Profile Plot attributes
        self.xz_canvas = None
        self.xz_figure = None
        self.xz_axes = None
        self.xz_plot_z_limit = 10  # Default limit in cm, +/- this value
        self.xz_plot_y_sections = 4  # Default number of Y-sections
        self.xz_plot_use_sections = True  # Default to showing the sectioned plot
        self.xz_z_min_ctrl = None
        self.xz_z_max_ctrl = None
        self.xz_plot_z_min = -10
        self.xz_plot_z_max = 10
        self.xz_plot_remove_outliers = True
        self.xz_plot_fit_method = "LOWESS" if statsmodels_available else "Polynomial"
        self.xz_plot_lowess_frac = 0.2
        self.fit_method_choice = None
        self.xz_plot_outlier_iqr_factor = 1.5
        self.xz_remove_outliers_button = None
        self.xz_plot_min_x_span_m = 1.5

        # XZ plot X-axis limit attributes
        self.xz_plot_fix_x_axis = False
        self.xz_plot_x_min = -2.0
        self.xz_plot_x_max = 2.0
        self.xz_fix_x_axis_button = None
        self.xz_x_min_ctrl = None
        self.xz_x_max_ctrl = None

        # XZ plot state management for undoing axis changes
        self.xz_plot_z_min_default, self.xz_plot_z_max_default = -10, 10
        self.xz_plot_x_min_default, self.xz_plot_x_max_default = -2.0, 2.0
        self.xz_plot_fix_x_axis_default = False
        self.xz_plot_z_min_prev, self.xz_plot_z_max_prev = -10, 10
        self.xz_plot_x_min_prev, self.xz_plot_x_max_prev = -2.0, 2.0
        self.xz_plot_fix_x_axis_prev = False
        self.xz_previous_settings_available = False
        self.xz_autoscale_button = None
        self.xz_defaults_button = None
        self.xz_previous_button = None
        self.xz_plot_section_gap_m = 0.02

        # 3D View Grid attributes
        self.show_grid = True
        self.grid_spacing = 1.0  # meters
        self.grid_visual = None
        self.show_grid_item = None

        # Point style attributes
        self.point_symbol = 'disc'
        self.symbol_choice = None

        # Labeling attributes
        self.labels = []
        self.label_buttons = {}

        # Image display attributes
        self.image_panel = None
        self.corresponding_image = None  # This will be a wx.Bitmap
        self.projected_image = None  # This will be the _p.jpg image

        # Projection attributes
        self.show_projection = False
        self.projection_button = None

        # Preloading and Caching attributes
        self.preload_cache = {}
        self.cache_lock = threading.Lock()
        self.CACHE_SIZE = 15  # Max items in cache (e.g., current +/- 7)
        self.preloading_threads = []

        self.InitUI()
        self.Maximize()
        self.Show()

        if filepath:
            wx.CallAfter(self.StartLoadingFile, filepath)

    def LoadSettings(self):
        """Loads settings, like custom labels, from a settings.json file."""
        settings_path = resource_path("settings.json")
        default_labels = ["Rutting", "Subsidence", "Potholes"]
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    # Safely load labels
                    loaded_labels = settings.get('labels', default_labels)
                    if isinstance(loaded_labels, list) and loaded_labels:
                        self.labels = [str(label) for label in loaded_labels]
                    else:
                        self.labels = default_labels

                    # Safely load XZ plot Z-limit
                    limit = settings.get('xz_plot_z_limit_cm')
                    if limit is not None and isinstance(limit, (int, float)):
                        self.xz_plot_z_limit = abs(limit)
                        # Also update the dynamic min/max values
                        self.xz_plot_z_min = -self.xz_plot_z_limit
                        self.xz_plot_z_max = self.xz_plot_z_limit
                    self.xz_plot_z_min_default = self.xz_plot_z_min
                    self.xz_plot_z_max_default = self.xz_plot_z_max
                    self.xz_plot_z_min_prev = self.xz_plot_z_min
                    self.xz_plot_z_max_prev = self.xz_plot_z_max

                    # Safely load number of Y sections for XZ plot
                    sections = settings.get('xz_plot_y_sections')
                    if sections is not None and isinstance(sections, int) and sections > 0:
                        self.xz_plot_y_sections = sections

                    # Safely load outlier removal settings
                    remove_outliers = settings.get('xz_plot_remove_outliers')
                    if isinstance(remove_outliers, bool):
                        self.xz_plot_remove_outliers = remove_outliers

                    iqr_factor = settings.get('xz_plot_outlier_iqr_factor')
                    if iqr_factor is not None and isinstance(iqr_factor, (int, float)) and iqr_factor > 0:
                        self.xz_plot_outlier_iqr_factor = iqr_factor

                    # Safely load fit method settings
                    fit_method = settings.get('xz_plot_fit_method', "LOWESS")
                    if fit_method == "LOWESS" and not statsmodels_available:
                        print(
                            "Warning: 'LOWESS' fit method selected but statsmodels is not installed. Falling back to 'Polynomial'.")
                        self.xz_plot_fit_method = "Polynomial"
                    else:
                        self.xz_plot_fit_method = fit_method

                    # Safely load min x-span for XZ plot
                    min_span = settings.get('xz_plot_min_x_span_m')
                    if min_span is not None and isinstance(min_span, (int, float)) and min_span >= 0:
                        self.xz_plot_min_x_span_m = min_span

                    # Safely load X-axis limit settings
                    self.xz_plot_fix_x_axis = bool(settings.get('xz_plot_fix_x_axis', False))
                    x_limit = settings.get('xz_plot_x_limit_m')
                    if x_limit is not None and isinstance(x_limit, (int, float)):
                        self.xz_plot_x_min = -abs(x_limit)
                        self.xz_plot_x_max = abs(x_limit)
                    self.xz_plot_fix_x_axis_default = self.xz_plot_fix_x_axis
                    self.xz_plot_x_min_default = self.xz_plot_x_min
                    self.xz_plot_x_max_default = self.xz_plot_x_max
                    self.xz_plot_fix_x_axis_prev = self.xz_plot_fix_x_axis
                    self.xz_plot_x_min_prev = self.xz_plot_x_min
                    self.xz_plot_x_max_prev = self.xz_plot_x_max

                    # Safely load section gap setting
                    gap_m = settings.get('xz_plot_section_gap_m')
                    if gap_m is not None and isinstance(gap_m, (int, float)) and gap_m > 0:
                        self.xz_plot_section_gap_m = gap_m

                    # Safely load grid settings
                    self.show_grid = bool(settings.get('view_show_grid', True))
                    grid_spacing = settings.get('view_grid_spacing_m')
                    if grid_spacing is not None and isinstance(grid_spacing, (int, float)) and grid_spacing > 0:
                        self.grid_spacing = grid_spacing

                    self.xz_plot_lowess_frac = float(settings.get('xz_plot_lowess_frac', 0.2))

            else:
                self.labels = default_labels
        except (json.JSONDecodeError, Exception) as e:
            self.labels = default_labels
            self.xz_plot_z_limit = 10
            self.xz_plot_z_min = -10
            self.xz_plot_z_max = 10
            self.xz_plot_y_sections = 4
            self.xz_plot_remove_outliers = True  # Keep this for consistency
            self.xz_plot_fit_method = "LOWESS" if statsmodels_available else "Polynomial"
            self.xz_plot_min_x_span_m = 1.5
            self.xz_plot_fix_x_axis = False
            self.xz_plot_x_min = -2.0
            self.xz_plot_x_max = 2.0
            self.xz_plot_z_min_default, self.xz_plot_z_max_default = self.xz_plot_z_min, self.xz_plot_z_max
            self.xz_plot_x_min_default, self.xz_plot_x_max_default = self.xz_plot_x_min, self.xz_plot_x_max
            self.xz_plot_fix_x_axis_default = self.xz_plot_fix_x_axis
            self.xz_plot_section_gap_m = 0.02
            self.show_grid = True
            self.grid_spacing = 1.0
            self.xz_plot_min_x_span_m = 1.5
            self.xz_plot_lowess_frac = 0.2
            print(f"Error loading settings.json: {e}. Using default labels and plot settings.")

    def InitUI(self):
        self.LoadSettings()
        self.SetTitle("Point Cloud Viewer | InViLab - UAntwerpen")
        try:
            icon_path = resource_path("genpycam.ico")
            if os.path.exists(icon_path):
                self.SetIcon(wx.Icon(icon_path, wx.BITMAP_TYPE_ICO))
        except Exception as e:
            print(f"Icon not found or failed to load: {e}")

        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        open_item = fileMenu.Append(wx.ID_OPEN, '&Open\tCtrl+O')
        exit_item = fileMenu.Append(wx.ID_EXIT, '&Exit\tAlt+F4')
        menubar.Append(fileMenu, '&File')

        viewMenu = wx.Menu()
        reset_view_item = viewMenu.Append(wx.ID_ANY, "Reset View\tCtrl+0")
        viewMenu.AppendSeparator()
        self.toggle_colors_item = viewMenu.Append(wx.ID_ANY, "Show Colors\tCtrl+C",
                                                  "Toggle point cloud colors", kind=wx.ITEM_CHECK)
        self.toggle_colors_item.Enable(False)
        self.color_by_z_item = viewMenu.Append(wx.ID_ANY, "Color by Z-Height\tCtrl+H",
                                               "Color points based on their Z coordinate", kind=wx.ITEM_CHECK)
        self.color_by_z_item.Enable(False)
        self.flip_z_item = viewMenu.Append(wx.ID_ANY, "Flip Z Axis\tCtrl+Z",
                                           "Flip the Z axis of the point cloud", kind=wx.ITEM_CHECK)
        self.flip_z_item.Enable(False)

        self.clip_by_y_item = viewMenu.Append(wx.ID_ANY, "Clip Y Axis",
                                              "Clip points based on their Y coordinate", kind=wx.ITEM_CHECK)
        self.clip_by_y_item.Enable(False)

        self.clip_by_index_item = viewMenu.Append(wx.ID_ANY, "Clip by Index",
                                                  "Clip points based on their index", kind=wx.ITEM_CHECK)
        self.clip_by_index_item.Enable(False)

        viewMenu.AppendSeparator()
        set_color_item = viewMenu.Append(wx.ID_ANY, "Set Point Color...",
                                         "Choose a color for points without color data")
        self.dark_bg_item = viewMenu.Append(wx.ID_ANY, "Dark Background", "Toggle dark background", kind=wx.ITEM_CHECK)

        # Add grid toggle to View menu
        self.show_grid_item = viewMenu.Append(wx.ID_ANY, "Show Grid\tCtrl+G",
                                              "Toggle visibility of the ground grid", kind=wx.ITEM_CHECK)
        self.show_grid_item.Check(self.show_grid)

        viewMenu.AppendSeparator()
        self.calculate_normals_item = viewMenu.Append(wx.ID_ANY, "Calculate Normals",
                                                      "Estimate normals for the point cloud")
        self.calculate_normals_item.Enable(False)
        self.show_normals_item = viewMenu.Append(wx.ID_ANY, "Show Normals", "Toggle visibility of normals",
                                                 kind=wx.ITEM_CHECK)
        self.show_normals_item.Enable(False)

        menubar.Append(viewMenu, "&View")
        self.SetMenuBar(menubar)

        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left panel for XZ profile plot
        xz_panel = wx.Panel(self.panel)
        xz_sizer = wx.BoxSizer(wx.VERTICAL)
        self.xz_figure = Figure(figsize=(4, 6), dpi=100)
        self.xz_axes = self.xz_figure.add_subplot(111)
        self.xz_canvas = FigureCanvas(xz_panel, -1, self.xz_figure)
        self.xz_canvas.Enable(False)
        xz_sizer.Add(self.xz_canvas, 1, wx.EXPAND | wx.ALL, 5)

        # XZ plot control buttons
        plot_controls_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.xz_plot_sections_button = wx.ToggleButton(xz_panel, label="Y-Sections")
        self.xz_plot_sections_button.SetValue(self.xz_plot_use_sections)
        self.xz_plot_sections_button.Enable(False)
        plot_controls_sizer.Add(self.xz_plot_sections_button, 1, wx.EXPAND | wx.RIGHT, 2)

        self.xz_remove_outliers_button = wx.ToggleButton(xz_panel, label="Remove Outliers")
        self.xz_remove_outliers_button.SetValue(self.xz_plot_remove_outliers)
        self.xz_remove_outliers_button.Enable(False)
        plot_controls_sizer.Add(self.xz_remove_outliers_button, 1, wx.EXPAND | wx.LEFT, 2)
        xz_sizer.Add(plot_controls_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # XZ plot axis control buttons
        axis_controls_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.xz_autoscale_button = wx.Button(xz_panel, label="Autoscale")
        self.xz_autoscale_button.Enable(False)
        axis_controls_sizer.Add(self.xz_autoscale_button, 1, wx.EXPAND | wx.RIGHT, 2)

        self.xz_defaults_button = wx.Button(xz_panel, label="Defaults")
        self.xz_defaults_button.Enable(False)
        axis_controls_sizer.Add(self.xz_defaults_button, 1, wx.EXPAND | wx.RIGHT, 2)

        self.xz_previous_button = wx.Button(xz_panel, label="Previous")
        self.xz_previous_button.Enable(False)
        axis_controls_sizer.Add(self.xz_previous_button, 1, wx.EXPAND)
        xz_sizer.Add(axis_controls_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Dropdown for selecting the profile fitting method
        fit_method_sizer = wx.BoxSizer(wx.HORIZONTAL)
        fit_method_sizer.Add(wx.StaticText(xz_panel, label="Fit Method:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        fit_choices = ["Polynomial"]
        if statsmodels_available:
            fit_choices.append("LOWESS")
        self.fit_method_choice = wx.Choice(xz_panel, choices=fit_choices)
        self.fit_method_choice.SetStringSelection(self.xz_plot_fit_method)
        self.fit_method_choice.Enable(False)
        fit_method_sizer.Add(self.fit_method_choice, 1, wx.EXPAND)
        xz_sizer.Add(fit_method_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Text controls for XZ plot Z-axis range for precision
        z_range_sizer = wx.BoxSizer(wx.HORIZONTAL)
        z_range_sizer.Add(wx.StaticText(xz_panel, label="Z-Range (cm):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        self.xz_z_min_ctrl = wx.TextCtrl(xz_panel, value=str(self.xz_plot_z_min), size=(50, -1),
                                         style=wx.TE_PROCESS_ENTER)
        self.xz_z_min_ctrl.Enable(False)
        z_range_sizer.Add(self.xz_z_min_ctrl, 1, wx.EXPAND | wx.RIGHT, 2)

        z_range_sizer.Add(wx.StaticText(xz_panel, label="to"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 2)

        self.xz_z_max_ctrl = wx.TextCtrl(xz_panel, value=str(self.xz_plot_z_max), size=(50, -1),
                                         style=wx.TE_PROCESS_ENTER)
        self.xz_z_max_ctrl.Enable(False)
        z_range_sizer.Add(self.xz_z_max_ctrl, 1, wx.EXPAND)
        xz_sizer.Add(z_range_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Controls for XZ plot X-axis range
        self.xz_fix_x_axis_button = wx.ToggleButton(xz_panel, label="Fix X-Axis")
        self.xz_fix_x_axis_button.SetValue(self.xz_plot_fix_x_axis)
        self.xz_fix_x_axis_button.Enable(False)
        xz_sizer.Add(self.xz_fix_x_axis_button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        x_range_sizer = wx.BoxSizer(wx.HORIZONTAL)
        x_range_sizer.Add(wx.StaticText(xz_panel, label="X-Range (m):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        self.xz_x_min_ctrl = wx.TextCtrl(xz_panel, value=str(self.xz_plot_x_min), size=(50, -1),
                                         style=wx.TE_PROCESS_ENTER)
        self.xz_x_min_ctrl.Enable(self.xz_plot_fix_x_axis)
        x_range_sizer.Add(self.xz_x_min_ctrl, 1, wx.EXPAND | wx.RIGHT, 2)

        x_range_sizer.Add(wx.StaticText(xz_panel, label="to"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 2)

        self.xz_x_max_ctrl = wx.TextCtrl(xz_panel, value=str(self.xz_plot_x_max), size=(50, -1),
                                         style=wx.TE_PROCESS_ENTER)
        self.xz_x_max_ctrl.Enable(self.xz_plot_fix_x_axis)
        x_range_sizer.Add(self.xz_x_max_ctrl, 1, wx.EXPAND)
        xz_sizer.Add(x_range_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        xz_panel.SetSizer(xz_sizer)
        main_sizer.Add(xz_panel, 1, wx.EXPAND)

        self.canvas = scene.SceneCanvas(parent=self.panel, keys='interactive', show=True, bgcolor='lightpink')
        self.view = self.canvas.central_widget.add_view()
        # Use an ArcballCamera for intuitive rotation and stable clipping.
        self.view.camera = cameras.ArcballCamera(fov=45, distance=10, up='+z')

        # Bind key handler directly to the VisPy canvas widget to capture navigation keys.
        self.canvas.native.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

        scene.visuals.XYZAxis(parent=self.view.scene)
        self.markers = scene.visuals.Markers(parent=self.view.scene)

        # Create the grid visual as a Line visual.
        self.grid_visual = Line(parent=self.view.scene, method='gl', width=1)
        self.grid_visual.visible = False

        self.normals_visual = Line(parent=self.view.scene, method='gl', width=1)
        self.normals_visual.visible = False

        controls_text = (
            "Controls:\n"
            "LMB: Orbit\n"
            "RMB / Scroll: Zoom\n"
            "Shift + LMB: Pan\n"
            "Shift + RMB: FOV\n"
            "ctrl + h: Show histogram colors"
            "ctrl + c: Show RGB colors\n"
            "'n' / 'p': Next/Prev File\n"
            "'s': Skip 10 Files"
        )
        self.controls_overlay = Text(
            controls_text, color='black', font_size=10, anchor_x='right',
            anchor_y='bottom', parent=self.canvas.scene
        )

        self.logo_overlay = None
        logo_path = resource_path("Logo_InViLab_2023_Horizontal_color.png")
        if os.path.exists(logo_path):
            try:
                pil_img = Image.open(logo_path).convert("RGBA")
                base_width = 150
                w_percent = (base_width / float(pil_img.size[0]))
                h_size = int((float(pil_img.size[1]) * float(w_percent)))
                pil_img = pil_img.resize((base_width, h_size), Image.LANCZOS)
                logo_np = np.array(pil_img)

                from vispy.scene.visuals import Image as VispyImage
                self.logo_overlay = VispyImage(
                    logo_np, parent=self.canvas.scene, method='auto'
                )
                self.logo_overlay.transform = scene.transforms.STTransform()
                self._logo_size = (base_width, h_size)
            except Exception as e:
                print(f"Failed to load logo image: {e}")

        self.canvas.events.resize.connect(self.OnCanvasResize)
        self.OnCanvasResize(None)

        # Main UI layout: XZ plot on the left, 3D canvas in the middle, controls on the right.
        main_sizer.Add(self.canvas.native, 2, wx.EXPAND)

        controls_panel = wx.Panel(self.panel)
        controls_sizer = wx.BoxSizer(wx.VERTICAL)

        # Image Display Panel
        self.image_panel = wx.Panel(controls_panel, style=wx.BORDER_SUNKEN)
        self.image_panel.SetBackgroundColour(wx.BLACK)
        self.image_panel.SetMinSize((-1, 200))  # Give it some initial size
        controls_sizer.Add(self.image_panel, 1, wx.EXPAND | wx.ALL, 5)

        # Projection Button
        self.projection_button = wx.ToggleButton(controls_panel, label="Show Projected Image")
        self.projection_button.SetValue(self.show_projection)
        self.projection_button.Enable(False)
        controls_sizer.Add(self.projection_button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Histogram Plot
        self.hist_figure = Figure(figsize=(4, 3), dpi=100)
        self.hist_axes = self.hist_figure.add_subplot(111)
        self.hist_canvas = FigureCanvas(controls_panel, -1, self.hist_figure)
        self.hist_canvas.Enable(False)
        controls_sizer.Add(self.hist_canvas, 1, wx.EXPAND | wx.ALL, 5)

        # Point Size Slider
        ps_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ps_sizer.Add(wx.StaticText(controls_panel, label="Point Size:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.point_size_slider = wx.Slider(controls_panel, value=10, minValue=1, maxValue=20)
        self.point_size_slider.Enable(False)
        ps_sizer.Add(self.point_size_slider, 1, wx.EXPAND)
        controls_sizer.Add(ps_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Point Style Controls
        style_sizer = wx.BoxSizer(wx.HORIZONTAL)
        style_sizer.Add(wx.StaticText(controls_panel, label="Point Style:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.symbol_choice = wx.Choice(controls_panel, choices=['disc', 'square', 'diamond', 'cross'])
        self.symbol_choice.SetStringSelection('disc')
        self.symbol_choice.Enable(False)
        style_sizer.Add(self.symbol_choice, 1, wx.EXPAND | wx.RIGHT, 5)

        controls_sizer.Add(style_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Z-Min/Max Sliders for coloring
        z_min_sizer = wx.BoxSizer(wx.HORIZONTAL)
        z_min_sizer.Add(wx.StaticText(controls_panel, label="Z-Min:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.z_min_slider = wx.Slider(controls_panel, value=0, minValue=0, maxValue=1000)
        self.z_min_slider.Enable(False)
        z_min_sizer.Add(self.z_min_slider, 1, wx.EXPAND)
        controls_sizer.Add(z_min_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        z_max_sizer = wx.BoxSizer(wx.HORIZONTAL)
        z_max_sizer.Add(wx.StaticText(controls_panel, label="Z-Max:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.z_max_slider = wx.Slider(controls_panel, value=1000, minValue=0, maxValue=1000)
        self.z_max_slider.Enable(False)
        z_max_sizer.Add(self.z_max_slider, 1, wx.EXPAND)
        controls_sizer.Add(z_max_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Z-Scale Slider
        z_scale_sizer = wx.BoxSizer(wx.HORIZONTAL)
        z_scale_sizer.Add(wx.StaticText(controls_panel, label="Z-Scale:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        # Slider from 10 to 200, representing 1.0x to 20.0x scale
        self.z_scale_slider = wx.Slider(controls_panel, value=10, minValue=10, maxValue=200)
        self.z_scale_slider.Enable(False)
        z_scale_sizer.Add(self.z_scale_slider, 1, wx.EXPAND)
        controls_sizer.Add(z_scale_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Y-Clip Min/Max Sliders
        y_min_sizer = wx.BoxSizer(wx.HORIZONTAL)
        y_min_sizer.Add(wx.StaticText(controls_panel, label="Y-Clip Min:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.y_min_slider = wx.Slider(controls_panel, value=0, minValue=0, maxValue=1000)
        self.y_min_slider.Enable(False)
        y_min_sizer.Add(self.y_min_slider, 1, wx.EXPAND)
        controls_sizer.Add(y_min_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        y_max_sizer = wx.BoxSizer(wx.HORIZONTAL)
        y_max_sizer.Add(wx.StaticText(controls_panel, label="Y-Clip Max:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.y_max_slider = wx.Slider(controls_panel, value=1000, minValue=0, maxValue=1000)
        self.y_max_slider.Enable(False)
        y_max_sizer.Add(self.y_max_slider, 1, wx.EXPAND)
        controls_sizer.Add(y_max_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Index Clip Min/Max Sliders
        index_min_sizer = wx.BoxSizer(wx.HORIZONTAL)
        index_min_sizer.Add(wx.StaticText(controls_panel, label="Index Min:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
                            5)
        self.index_min_slider = wx.Slider(controls_panel, value=0, minValue=0, maxValue=1000)
        self.index_min_slider.Enable(False)
        index_min_sizer.Add(self.index_min_slider, 1, wx.EXPAND)
        controls_sizer.Add(index_min_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        index_max_sizer = wx.BoxSizer(wx.HORIZONTAL)
        index_max_sizer.Add(wx.StaticText(controls_panel, label="Index Max:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
                            5)
        self.index_max_slider = wx.Slider(controls_panel, value=1000, minValue=0, maxValue=1000)
        self.index_max_slider.Enable(False)
        index_max_sizer.Add(self.index_max_slider, 1, wx.EXPAND)
        controls_sizer.Add(index_max_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Labeling Buttons
        label_box = wx.StaticBox(controls_panel, label="Label Data")
        # Use a grid sizer for flexible layout
        label_grid_sizer = wx.GridSizer(rows=0, cols=2, vgap=2, hgap=2)

        for label_text in self.labels:
            button = wx.ToggleButton(controls_panel, label=label_text)
            button.Enable(False)
            self.label_buttons[label_text] = button
            label_grid_sizer.Add(button, 0, wx.EXPAND)

        # Wrap the grid sizer in a static box sizer
        label_sizer = wx.StaticBoxSizer(label_box, wx.VERTICAL)
        label_sizer.Add(label_grid_sizer, 1, wx.EXPAND | wx.ALL, 2)

        controls_sizer.Add(label_sizer, 0, wx.EXPAND | wx.ALL, 5)

        controls_panel.SetSizer(controls_sizer)
        main_sizer.Add(controls_panel, 1, wx.EXPAND | wx.ALL, 5)

        self.panel.SetSizer(main_sizer)
        self.CreateStatusBar()

        self.Bind(wx.EVT_MENU, self.OnOpen, open_item)
        self.Bind(wx.EVT_MENU, self.OnExit, exit_item)
        self.Bind(wx.EVT_MENU, self.OnResetView, reset_view_item)
        self.Bind(wx.EVT_MENU, self.OnToggleColors, self.toggle_colors_item)
        self.Bind(wx.EVT_MENU, self.OnColorByZ, self.color_by_z_item)
        self.Bind(wx.EVT_MENU, self.OnClipByY, self.clip_by_y_item)
        self.Bind(wx.EVT_MENU, self.OnClipByIndex, self.clip_by_index_item)
        self.Bind(wx.EVT_MENU, self.OnFlipZ, self.flip_z_item)
        self.Bind(wx.EVT_SLIDER, self.OnPointSizeChange, self.point_size_slider)
        self.Bind(wx.EVT_SLIDER, self.OnZSliderChange, self.z_min_slider)
        self.Bind(wx.EVT_SLIDER, self.OnZSliderChange, self.z_max_slider)
        self.Bind(wx.EVT_SLIDER, self.OnZScaleSliderChange, self.z_scale_slider)
        self.Bind(wx.EVT_SLIDER, self.OnYSliderChange, self.y_min_slider)
        self.Bind(wx.EVT_SLIDER, self.OnYSliderChange, self.y_max_slider)
        self.Bind(wx.EVT_SLIDER, self.OnIndexSliderChange, self.index_min_slider)
        self.Bind(wx.EVT_SLIDER, self.OnIndexSliderChange, self.index_max_slider)
        self.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleXZSections, self.xz_plot_sections_button)
        self.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleRemoveOutliers, self.xz_remove_outliers_button)
        self.Bind(wx.EVT_CHOICE, self.OnFitMethodChange, self.fit_method_choice)

        # Bindings for XZ axis control buttons
        self.Bind(wx.EVT_BUTTON, self.OnXZAutoscale, self.xz_autoscale_button)
        self.Bind(wx.EVT_BUTTON, self.OnXZResetToDefaults, self.xz_defaults_button)
        self.Bind(wx.EVT_BUTTON, self.OnXZRevertToPrevious, self.xz_previous_button)

        # Bindings for XZ Range Text Controls
        self.Bind(wx.EVT_TEXT_ENTER, self.OnXZRangeChange, self.xz_z_min_ctrl)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnXZRangeChange, self.xz_z_max_ctrl)
        self.xz_z_min_ctrl.Bind(wx.EVT_KILL_FOCUS, self.OnXZRangeChange)
        self.xz_z_max_ctrl.Bind(wx.EVT_KILL_FOCUS, self.OnXZRangeChange)

        # Bindings for XZ X-Range Controls
        self.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleFixXAxis, self.xz_fix_x_axis_button)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnXRangeChange, self.xz_x_min_ctrl)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnXRangeChange, self.xz_x_max_ctrl)
        self.xz_x_min_ctrl.Bind(wx.EVT_KILL_FOCUS, self.OnXRangeChange)
        self.xz_x_max_ctrl.Bind(wx.EVT_KILL_FOCUS, self.OnXRangeChange)

        # Bind Labeling Buttons
        for button in self.label_buttons.values():
            self.Bind(wx.EVT_TOGGLEBUTTON, self.OnLabelToggle, button)
        self.Bind(wx.EVT_MENU, self.OnCalculateNormals, self.calculate_normals_item)
        self.Bind(wx.EVT_MENU, self.OnShowNormals, self.show_normals_item)
        self.Bind(wx.EVT_MENU, self.OnToggleBackground, self.dark_bg_item)
        self.Bind(wx.EVT_MENU, self.OnSetPointColor, set_color_item)
        # Bind grid toggle
        self.Bind(wx.EVT_MENU, self.OnToggleGrid, self.show_grid_item)

        # Bind point style controls
        self.Bind(wx.EVT_CHOICE, self.OnSymbolChange, self.symbol_choice)

        # Bind projection button
        self.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleProjection, self.projection_button)

        self.image_panel.Bind(wx.EVT_PAINT, self.OnPaintImage)
        self.image_panel.Bind(wx.EVT_SIZE, self.OnSizeImage)

    def OnKeyDown(self, event):
        """Handle key down events for navigation."""
        # This print statement helps debug; it will appear in your console.
        # print(f"Key press detected. KeyCode: {event.GetKeyCode()}")
        keycode = event.GetKeyCode()

        if not self.file_list:
            event.Skip()
            return

        new_index = self.current_file_index

        if keycode == ord('N'):
            new_index = (self.current_file_index + 1) % len(self.file_list)
        elif keycode == ord('P'):
            new_index = (self.current_file_index - 1) % len(self.file_list)
        elif keycode == ord('S'):
            new_index = (self.current_file_index + 10) % len(self.file_list)
        else:
            event.Skip()
            return

        if new_index == self.current_file_index:
            return

        # Check if the requested file is in the cache
        new_filepath = os.path.join(self.current_dir, self.file_list[new_index])
        with self.cache_lock:
            cached_data = self.preload_cache.pop(new_filepath, None)

        if cached_data:
            # If found in cache, load it directly without a thread
            print(f"Loading {os.path.basename(new_filepath)} from cache.")
            self.current_file_index = new_index
            self.OnLoadComplete(*cached_data)
        else:
            # If not in cache, load it the normal way
            self.StartLoadingFile(new_filepath)

    def OnCanvasResize(self, event):
        if not self.canvas: return
        w, h = self.canvas.size
        self.controls_overlay.pos = (w - 10, 10)
        if self.logo_overlay is not None:
            img_w, img_h = self._logo_size
            margin = 10
            self.logo_overlay.transform.translate = (img_w // 2 + margin, img_h // 2 + margin, 0)

    def OnOpen(self, event):
        wildcard = "Point Cloud Files (*.pcd, *.ply)|*.pcd;*.ply|All files (*.*)|*.*"
        with wx.FileDialog(self, "Open Point Cloud file", wildcard=wildcard,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            filepath = fd.GetPath()

            # On manual open, clear the cache and stop any preloading
            self._clear_preloader()
            with self.cache_lock:
                self.preload_cache.clear()

            # Check if the file is in the cache (it shouldn't be, but for safety)
            cached_data = self.preload_cache.pop(filepath, None)
            if not cached_data:
                self.StartLoadingFile(filepath)


    def OnPointSizeChange(self, event):
        self.UpdateDisplay()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnResetView(self, event=None):
        if self.view and self.points is not None and len(self.points) > 0:
            # Use a scaled copy for calculating camera position
            pts = self.points.copy()
            if self.z_scale_factor != 1.0:
                # Scale relative to the mean to keep the cloud centered
                z_mean = np.mean(pts[:, 2])
                pts[:, 2] = z_mean + (pts[:, 2] - z_mean) * self.z_scale_factor

            center = np.mean(pts, axis=0)
            extent = np.ptp(pts, axis=0)
            max_extent = np.max(extent)
            distance = max(max_extent * 2.0, 1.0)  # Give a bit more room
            self.view.camera.center = center
            self.view.camera.distance = distance
            self.view.camera.azimuth = 45
            self.view.camera.elevation = 30

            # Manually set clipping planes for better rendering.
            # This prevents points from disappearing when rotating the camera.
            self.view.camera.near = distance / 1000.0
            self.view.camera.far = distance + max_extent * 2.0
        elif self.view:
            self.view.camera.reset()
            self.view.camera.distance = 10

    def OnToggleColors(self, event):
        self.show_colors = event.IsChecked()
        # Ensure color modes are mutually exclusive
        if self.show_colors:
            self.color_by_z = False
            self.color_by_z_item.Check(False)
            self.z_min_slider.Enable(False)
            self.z_max_slider.Enable(False)
            self.hist_canvas.Enable(False)
        self.UpdateDisplay()

    def OnColorByZ(self, event):
        self.color_by_z = event.IsChecked()
        is_enabled = self.color_by_z and self.points is not None
        self.z_min_slider.Enable(is_enabled)
        self.z_max_slider.Enable(is_enabled)
        self.hist_canvas.Enable(is_enabled)

        # Ensure color modes are mutually exclusive
        if self.color_by_z:
            self.show_colors = False
            self.toggle_colors_item.Check(False)
        self.UpdateDisplay()

    def OnClipByY(self, event):
        self.clip_by_y = event.IsChecked()
        is_enabled = self.clip_by_y and self.points is not None
        self.y_min_slider.Enable(is_enabled)
        self.y_max_slider.Enable(is_enabled)
        self.UpdateDisplay()

    def OnClipByIndex(self, event):
        self.clip_by_index = event.IsChecked()
        is_enabled = self.clip_by_index and self.points is not None
        self.index_min_slider.Enable(is_enabled)
        self.index_max_slider.Enable(is_enabled)
        self.UpdateDisplay()

    def OnZSliderChange(self, event):
        """Handles changes from the Z-Min and Z-Max sliders."""
        min_slider_val = self.z_min_slider.GetValue()
        max_slider_val = self.z_max_slider.GetValue()

        if min_slider_val > max_slider_val:
            self.z_min_slider.SetValue(max_slider_val)
            min_slider_val = max_slider_val

        data_range = self.z_data_max - self.z_data_min
        self.z_color_min = self.z_data_min + (min_slider_val / 1000.0) * data_range
        self.z_color_max = self.z_data_min + (max_slider_val / 1000.0) * data_range

        self.UpdateDisplay()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnYSliderChange(self, event):
        """Handles changes from the Y-Clip Min and Max sliders."""
        min_slider_val = self.y_min_slider.GetValue()
        max_slider_val = self.y_max_slider.GetValue()

        if min_slider_val > max_slider_val:
            self.y_min_slider.SetValue(max_slider_val)
            min_slider_val = max_slider_val

        data_range = self.y_data_max - self.y_data_min
        if data_range > 1e-6:
            self.y_clip_min = self.y_data_min + (min_slider_val / 1000.0) * data_range
            self.y_clip_max = self.y_data_min + (max_slider_val / 1000.0) * data_range

        self.UpdateDisplay()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnIndexSliderChange(self, event):
        """Handles changes from the Index-Clip Min and Max sliders."""
        min_slider_val = self.index_min_slider.GetValue()
        max_slider_val = self.index_max_slider.GetValue()

        if min_slider_val > max_slider_val:
            self.index_min_slider.SetValue(max_slider_val)
            min_slider_val = max_slider_val

        # The sliders go from 0 to 1000. We map this to the index range.
        # Ensure index_data_max is at least 0 to avoid negative ranges
        max_index = max(0, self.index_data_max)
        self.index_clip_min = int((min_slider_val / 1000.0) * max_index)
        self.index_clip_max = int((max_slider_val / 1000.0) * max_index)

        self.UpdateDisplay()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnZScaleSliderChange(self, event):
        """Handles changes from the Z-Scale slider."""
        slider_val = self.z_scale_slider.GetValue()
        # Map slider value (10-200) to scale factor (1.0-20.0)
        self.z_scale_factor = slider_val / 10.0
        self.UpdateDisplay()
        if self.canvas:
            self.canvas.native.SetFocus()

    def _store_previous_xz_settings(self):
        """Saves the current XZ plot axis settings to the 'previous' state."""
        self.xz_plot_z_min_prev = self.xz_plot_z_min
        self.xz_plot_z_max_prev = self.xz_plot_z_max
        self.xz_plot_x_min_prev = self.xz_plot_x_min
        self.xz_plot_x_max_prev = self.xz_plot_x_max
        self.xz_plot_fix_x_axis_prev = self.xz_plot_fix_x_axis
        self.xz_previous_settings_available = True
        if self.xz_previous_button:
            self.xz_previous_button.Enable(True)

    def OnXZRangeChange(self, event):
        """Handles changes from the XZ-Plot Z-Min and Z-Max text controls."""
        self._store_previous_xz_settings()
        try:
            min_val = float(self.xz_z_min_ctrl.GetValue())
            max_val = float(self.xz_z_max_ctrl.GetValue())
        except ValueError:
            # If input is not a valid number, revert to the last known good values
            self.xz_z_min_ctrl.SetValue(f"{self.xz_plot_z_min:.2f}")
            self.xz_z_max_ctrl.SetValue(f"{self.xz_plot_z_max:.2f}")
            print("Invalid input for Z-Range. Please enter numbers only.")
            # Allow focus to change away from the bad input, but consume Enter presses.
            if isinstance(event, wx.FocusEvent):
                event.Skip()
            return

        if min_val > max_val:
            # Swap if min is greater than max
            min_val, max_val = max_val, min_val

        self.xz_plot_z_min = min_val
        self.xz_plot_z_max = max_val

        # Update controls to reflect validated/swapped values, formatted nicely
        self.xz_z_min_ctrl.SetValue(f"{self.xz_plot_z_min:.2f}")
        self.xz_z_max_ctrl.SetValue(f"{self.xz_plot_z_max:.2f}")

        self.UpdateXZPlot()
        # Allow focus to change for EVT_KILL_FOCUS, but consume EVT_TEXT_ENTER
        if isinstance(event, wx.FocusEvent):
            event.Skip()

    def OnToggleFixXAxis(self, event):
        """Toggles fixing the X-axis on the XZ profile plot."""
        self._store_previous_xz_settings()
        self.xz_plot_fix_x_axis = event.IsChecked()
        self.xz_x_min_ctrl.Enable(self.xz_plot_fix_x_axis)
        self.xz_x_max_ctrl.Enable(self.xz_plot_fix_x_axis)
        self.UpdateXZPlot()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnXRangeChange(self, event):
        """Handles changes from the XZ-Plot X-Min and X-Max text controls."""
        self._store_previous_xz_settings()
        try:
            min_val = float(self.xz_x_min_ctrl.GetValue())
            max_val = float(self.xz_x_max_ctrl.GetValue())
        except ValueError:
            self.xz_x_min_ctrl.SetValue(f"{self.xz_plot_x_min:.2f}")
            self.xz_x_max_ctrl.SetValue(f"{self.xz_plot_x_max:.2f}")
            print("Invalid input for X-Range. Please enter numbers only.")
            # Allow focus to change away from the bad input, but consume Enter presses.
            if isinstance(event, wx.FocusEvent):
                event.Skip()
            return

        if min_val > max_val:
            min_val, max_val = max_val, min_val

        self.xz_plot_x_min = min_val
        self.xz_plot_x_max = max_val

        self.xz_x_min_ctrl.SetValue(f"{self.xz_plot_x_min:.2f}")
        self.xz_x_max_ctrl.SetValue(f"{self.xz_plot_x_max:.2f}")

        # Only update if the axis is fixed
        if self.xz_plot_fix_x_axis:
            self.UpdateXZPlot()
        # Allow focus to change for EVT_KILL_FOCUS, but consume EVT_TEXT_ENTER
        if isinstance(event, wx.FocusEvent):
            event.Skip()

    def _update_xz_controls_from_state(self):
        """Updates the XZ plot control widgets to match the current state attributes."""
        self.xz_z_min_ctrl.SetValue(f"{self.xz_plot_z_min:.2f}")
        self.xz_z_max_ctrl.SetValue(f"{self.xz_plot_z_max:.2f}")
        self.xz_x_min_ctrl.SetValue(f"{self.xz_plot_x_min:.2f}")
        self.xz_x_max_ctrl.SetValue(f"{self.xz_plot_x_max:.2f}")
        self.xz_fix_x_axis_button.SetValue(self.xz_plot_fix_x_axis)
        self.xz_x_min_ctrl.Enable(self.xz_plot_fix_x_axis)
        self.xz_x_max_ctrl.Enable(self.xz_plot_fix_x_axis)

    def OnXZAutoscale(self, event):
        """Autoscales the XZ plot to fit the currently visible data."""
        if self.points is None:
            return

        # Get the currently visible points, same logic as in UpdateXZPlot
        visible_mask = np.ones(len(self.points), dtype=bool)
        if self.clip_by_y:
            visible_mask &= (self.points[:, 1] >= self.y_clip_min) & (self.points[:, 1] <= self.y_clip_max)
        if self.clip_by_index:
            indices = np.arange(len(self.points))
            visible_mask &= (indices >= self.index_clip_min) & (indices <= self.index_clip_max)
        points_to_show = self.points[visible_mask]

        if len(points_to_show) < 2:
            return

        self._store_previous_xz_settings()

        x_coords = points_to_show[:, 0]
        z_coords = -points_to_show[:, 2] / self.z_scale_factor * 100

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)

        # Add a 5% margin
        x_margin = (x_max - x_min) * 0.05 if (x_max - x_min) > 1e-6 else 0.1
        z_margin = (z_max - z_min) * 0.05 if (z_max - z_min) > 1e-6 else 0.1

        self.xz_plot_x_min, self.xz_plot_x_max = x_min - x_margin, x_max + x_margin
        self.xz_plot_z_min, self.xz_plot_z_max = z_min - z_margin, z_max + z_margin
        self.xz_plot_fix_x_axis = True  # Autoscaling implies fixing the axis to the new range

        self._update_xz_controls_from_state()
        self.UpdateXZPlot()

    def OnXZResetToDefaults(self, event):
        """Resets the XZ plot axes to their default values from settings."""
        self._store_previous_xz_settings()
        self.xz_plot_z_min, self.xz_plot_z_max = self.xz_plot_z_min_default, self.xz_plot_z_max_default
        self.xz_plot_x_min, self.xz_plot_x_max = self.xz_plot_x_min_default, self.xz_plot_x_max_default
        self.xz_plot_fix_x_axis = self.xz_plot_fix_x_axis_default
        self._update_xz_controls_from_state()
        self.UpdateXZPlot()

    def OnXZRevertToPrevious(self, event):
        """Reverts the XZ plot axes to their previous settings."""
        if not self.xz_previous_settings_available:
            return
        # Swap current and previous settings
        (self.xz_plot_z_min, self.xz_plot_z_min_prev) = (self.xz_plot_z_min_prev, self.xz_plot_z_min)
        (self.xz_plot_z_max, self.xz_plot_z_max_prev) = (self.xz_plot_z_max_prev, self.xz_plot_z_max)
        (self.xz_plot_x_min, self.xz_plot_x_min_prev) = (self.xz_plot_x_min_prev, self.xz_plot_x_min)
        (self.xz_plot_x_max, self.xz_plot_x_max_prev) = (self.xz_plot_x_max_prev, self.xz_plot_x_max)
        (self.xz_plot_fix_x_axis, self.xz_plot_fix_x_axis_prev) = (self.xz_plot_fix_x_axis_prev,
                                                                   self.xz_plot_fix_x_axis)
        self._update_xz_controls_from_state()
        self.UpdateXZPlot()

    def OnToggleXZSections(self, event):
        """Toggles between sectioned and global XZ profile plot."""
        self.xz_plot_use_sections = event.IsChecked()
        self.UpdateXZPlot()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnToggleRemoveOutliers(self, event):
        """Toggles outlier removal for the XZ profile plot."""
        self.xz_plot_remove_outliers = event.IsChecked()
        self.UpdateXZPlot()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnFitMethodChange(self, event):
        """Handles selection of a new profile fitting method."""
        self.xz_plot_fit_method = event.GetString()
        self.UpdateXZPlot()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnLabelToggle(self, event):
        """Handles toggling of a data label button."""
        if self.current_file_index == -1:
            return

        button = event.GetEventObject()
        label = button.GetLabel()
        is_active = button.GetValue()

        self.UpdateLabelInCSV(label, add=is_active)
        if self.canvas:
            self.canvas.native.SetFocus()

    def UpdateLabelInCSV(self, label_text, add):
        """Adds or removes a label entry for the current file in labels.csv."""
        if self.current_dir is None or not self.file_list:
            return

        label = label_text.lower()
        filename = self.file_list[self.current_file_index]
        labels_csv_path = os.path.join(self.current_dir, 'labels.csv')
        header = ['filename', 'label']
        entry_to_change = [filename, label]

        rows = []
        if os.path.exists(labels_csv_path):
            try:
                with open(labels_csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if row:
                            rows.append(row)
            except (StopIteration, IOError) as e:
                print(f"Warning: Could not read labels.csv: {e}")

        # Filter out the entry we are changing
        rows = [row for row in rows if row != entry_to_change]

        if add:
            rows.append(entry_to_change)

        try:
            with open(labels_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        except IOError as e:
            wx.LogError(f"Could not write to labels.csv in {self.current_dir}.\nError: {e}")

    def UpdateLabelButtonStates(self):
        """Reads labels.csv and sets the toggle state of the label buttons."""
        # Enable/disable all buttons and reset their state
        for btn in self.label_buttons.values():
            btn.Enable(self.points is not None)
            btn.SetValue(False)

        if self.points is None or self.current_dir is None:
            return

        # Get all labels for the current file from the CSV
        labels_for_file = set()
        labels_csv_path = os.path.join(self.current_dir, 'labels.csv')
        if not os.path.exists(labels_csv_path):
            return

        try:
            filename = self.file_list[self.current_file_index]
            with open(labels_csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) == 2 and row[0] == filename:
                        labels_for_file.add(row[1].lower())
        except (IOError, StopIteration, IndexError) as e:
            print(f"Warning: Could not process labels.csv: {e}")
            return
        for label_text, button in self.label_buttons.items():
            if label_text.lower() in labels_for_file:
                button.SetValue(True)

    def OnFlipZ(self, event):
        self.flip_z = event.IsChecked()
        self.UpdateDisplay()

    def OnToggleBackground(self, event):
        if event.IsChecked():
            self.canvas.bgcolor = 'black'
            self.controls_overlay.color = 'white'
        else:
            self.canvas.bgcolor = 'lightpink'
            self.controls_overlay.color = 'black'
        self.canvas.update()

    def OnToggleGrid(self, event):
        """Toggles the visibility of the 3D grid."""
        self.show_grid = event.IsChecked()
        # The grid will be updated in the next UpdateDisplay call,
        self.UpdateDisplay()

    def OnSymbolChange(self, event):
        """Handles changing the point symbol."""
        self.point_symbol = event.GetString()
        self.UpdateDisplay()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnSetPointColor(self, event):
        with wx.ColourDialog(self) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                color_data = dlg.GetColourData()
                color = color_data.GetColour()
                self.default_point_color = Color(color.GetAsString(wx.C2S_HTML_SYNTAX)).hex
                self.UpdateDisplay()

    def OnPaintImage(self, event):
        """Draws the corresponding image, scaled to fit the panel."""
        if not self.image_panel:
            return
        dc = wx.PaintDC(self.image_panel)

        # Get panel size and clear it
        panel_w, panel_h = self.image_panel.GetClientSize()
        dc.SetBrush(wx.Brush(wx.BLACK))
        dc.DrawRectangle(0, 0, panel_w, panel_h)

        # Decide which image to display
        image_to_display = None
        if self.show_projection and self.projected_image:
            image_to_display = self.projected_image
        elif self.corresponding_image:
            image_to_display = self.corresponding_image

        if image_to_display and image_to_display.IsOk():
            img_w, img_h = image_to_display.GetWidth(), image_to_display.GetHeight()

            if panel_w == 0 or panel_h == 0 or img_w == 0 or img_h == 0:
                return

            # Calculate scale to fit image while maintaining aspect ratio
            scale = min(panel_w / img_w, panel_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)

            # Center the image
            x_offset = (panel_w - new_w) // 2
            y_offset = (panel_h - new_h) // 2

            # Create a scaled bitmap
            image = image_to_display.ConvertToImage()
            image.Rescale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)
            scaled_bitmap = wx.Bitmap(image)

            dc.DrawBitmap(scaled_bitmap, x_offset, y_offset, useMask=False)

    def OnSizeImage(self, event):
        if self.image_panel:
            self.image_panel.Refresh()
        event.Skip()

    def OnToggleProjection(self, event):
        """Toggles projection of points onto the image."""
        self.show_projection = event.IsChecked()
        # Redraw the image panel
        self.image_panel.Refresh()
        if self.canvas:
            self.canvas.native.SetFocus()

    def UpdateDisplay(self):
        if self.points is None:
            self.markers.set_data(np.empty((0, 3)))
            if self.grid_visual:
                self.grid_visual.visible = False
            self.SetStatusText("No file loaded.")
            self.canvas.update()
            return

        point_size = self.point_size_slider.GetValue()

        # Use a single boolean mask for all filters
        # Start with a mask that includes all points
        total_points = len(self.points)
        visible_mask = np.ones(total_points, dtype=bool)

        # 1. Apply Y-Clipping
        if self.clip_by_y:
            y_coords = self.points[:, 1]
            visible_mask &= (y_coords >= self.y_clip_min) & (y_coords <= self.y_clip_max)

        # 2. Apply Index-Clipping
        if self.clip_by_index:
            indices = np.arange(total_points)
            visible_mask &= (indices >= self.index_clip_min) & (indices <= self.index_clip_max)

        points_to_show = self.points[visible_mask]
        colors_to_show = self.colors[visible_mask] if self.colors is not None else None

        # Update the histogram with the Z-data of the *visible* points
        self.UpdateHistogramPlot(z_data=points_to_show[:, 2])

        # Create a copy for all display modifications (scaling, flipping)
        display_points = points_to_show.copy()

        # Apply Z-Flipping first
        if self.flip_z:
            display_points[:, 2] *= -1

        # Apply Z-Scaling
        if self.z_scale_factor != 1.0:
            if len(display_points) > 0:
                z_mean = np.mean(display_points[:, 2])
                display_points[:, 2] = z_mean + (display_points[:, 2] - z_mean) * self.z_scale_factor

        # Determine colors for the visible points (using original Z values)
        display_colors = self.default_point_color
        if self.color_by_z and len(points_to_show) > 0:
            z_coords = points_to_show[:, 2]  # Use original z-coords for coloring
            z_range = self.z_color_max - self.z_color_min
            if z_range > 1e-6:  # Avoid division by zero for flat point clouds
                # Normalize z-coordinates to the user-defined range
                normalized_z = (z_coords - self.z_color_min) / z_range
                # Clip values to be within [0, 1] for the colormap
                clipped_z = np.clip(normalized_z, 0, 1)
                colormap = get_colormap('viridis')
                display_colors = colormap.map(clipped_z)
        elif self.show_colors and colors_to_show is not None:
            display_colors = colors_to_show

        self.markers.set_data(
            display_points,
            size=point_size,
            edge_width=0,  # Set to 0 for cleaner look, especially with shading
            face_color=display_colors,
            symbol=self.point_symbol
        )

        visible_count = len(points_to_show)
        total_count = len(self.points)
        status_text = f"Showing {visible_count} of {total_count} points"
        if self.file_list:
            status_text += f" ({self.current_file_index + 1}/{len(self.file_list)})"
        status_text += f" | Point Size: {point_size}"
        if self.color_by_z:
            status_text += f" | Z-Range: [{self.z_color_min:.2f}, {self.z_color_max:.2f}]"
        if self.clip_by_y:
            status_text += f" | Y-Clip: [{self.y_clip_min:.2f}, {self.y_clip_max:.2f}]"
        if self.clip_by_index:
            status_text += f" | Index: [{self.index_clip_min}, {self.index_clip_max}]"
        if self.z_scale_factor != 1.0:
            status_text += f" | Z-Scale: {self.z_scale_factor}"
        self.SetStatusText(status_text)

        # Update normals visual based on visible and scaled points
        if self.normals is not None and self.show_normals and len(points_to_show) > 0:
            # Get the normals corresponding to the visible points
            visible_normals = self.normals[visible_mask]

            # Copy normals for modification
            display_normals = visible_normals.copy()

            # Also flip the normals if Z is flipped
            if self.flip_z:
                display_normals[:, 2] *= -1

            # The normals should be drawn from the scaled point positions
            start_points = display_points

            # Calculate the length of the normal vectors
            # Use the scaled points' extent to keep the normals proportional
            extent = np.max(np.ptp(display_points, axis=0))
            normal_length = extent * 0.02

            # Define the line segments for the normals
            end_points = start_points + display_normals * normal_length

            line_verts = np.empty((len(start_points) * 2, 3), dtype=np.float32)
            line_verts[0::2] = start_points
            line_verts[1::2] = end_points

            self.normals_visual.set_data(pos=line_verts, color='cyan', connect='segments')
            self.normals_visual.visible = True
        else:
            self.normals_visual.visible = False

        self.UpdateXZPlot()
        # Update the grid based on the final visible points
        self.UpdateGrid(display_points)

        self.canvas.update()

    def UpdateGrid(self, display_points):
        """
        Generates and displays a finite, colored grid on the ground plane based on the
        extent of the currently visible points and their Y-sections.
        """
        if self.grid_visual is None or not self.show_grid or display_points is None or len(display_points) < 2:
            if self.grid_visual:
                self.grid_visual.visible = False
            return

        # Place grid at the bottom of the *displayed* point cloud
        z_pos = np.min(display_points[:, 2])

        # Get Y-section coloring info from ORIGINAL points
        # This ensures the grid coloring matches the XZ plot sections
        visible_mask = np.ones(len(self.points), dtype=bool)
        if self.clip_by_y:
            visible_mask &= (self.points[:, 1] >= self.y_clip_min) & (self.points[:, 1] <= self.y_clip_max)
        if self.clip_by_index:
            indices = np.arange(len(self.points))
            visible_mask &= (indices >= self.index_clip_min) & (indices <= self.index_clip_max)
        y_coords_all = self.points[visible_mask][:, 1]

        if len(y_coords_all) == 0:
            self.grid_visual.visible = False
            return

        y_min_all, y_max_all = np.min(y_coords_all), np.max(y_coords_all)
        y_range = y_max_all - y_min_all
        num_sections = self.xz_plot_y_sections if y_range > 1e-6 else 1
        section_width = y_range / num_sections if num_sections > 0 else 0
        section_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        def get_y_section_color(y_coord):
            if section_width <= 1e-6:
                return Color(section_colors[0])
            index = int((y_coord - y_min_all) / section_width)
            index = np.clip(index, 0, num_sections - 1)
            return Color(section_colors[index % len(section_colors)])

        # Calculate grid bounds from DISPLAYED points
        x_min, x_max = np.min(display_points[:, 0]), np.max(display_points[:, 0])
        y_min, y_max = np.min(display_points[:, 1]), np.max(display_points[:, 1])

        x_lines = np.arange(np.floor(x_min / self.grid_spacing) * self.grid_spacing,
                            np.ceil(x_max / self.grid_spacing) * self.grid_spacing + self.grid_spacing,
                            self.grid_spacing)
        y_lines = np.arange(np.floor(y_min / self.grid_spacing) * self.grid_spacing,
                            np.ceil(y_max / self.grid_spacing) * self.grid_spacing + self.grid_spacing,
                            self.grid_spacing)

        if len(x_lines) < 2 or len(y_lines) < 2:
            self.grid_visual.visible = False
            return

        vertices = []
        colors = []

        # Lines along Y axis (constant X), segmented and colored by Y-section
        for x in x_lines:
            for i in range(len(y_lines) - 1):
                y_start, y_end = y_lines[i], y_lines[i + 1]
                y_mid = (y_start + y_end) / 2.0
                segment_color = get_y_section_color(y_mid)
                vertices.extend([[x, y_start, z_pos], [x, y_end, z_pos]])
                colors.extend([segment_color.rgba, segment_color.rgba])

        for y in y_lines:  # Lines along X axis (constant Y) - section color
            line_color = get_y_section_color(y)
            vertices.extend([[x_lines[0], y, z_pos], [x_lines[-1], y, z_pos]])
            colors.extend([line_color.rgba, line_color.rgba])

        self.grid_visual.set_data(pos=np.array(vertices, dtype=np.float32),
                                  color=np.array(colors, dtype=np.float32),
                                  connect='segments')
        self.grid_visual.visible = True

    def UpdateXZPlot(self):
        """Creates or updates the X-Z profile plot in the main window."""
        if not self.xz_canvas:
            return

        if self.points is None or len(self.points) == 0:
            self.xz_axes.clear()
            self.xz_axes.text(0.5, 0.5, 'No points to display.', horizontalalignment='center',
                              verticalalignment='center')
            self.xz_canvas.draw()
            return

        # Get the currently visible points, including all filters
        total_points = len(self.points)
        visible_mask = np.ones(total_points, dtype=bool)

        if self.clip_by_y:
            y_coords = self.points[:, 1]
            visible_mask &= (y_coords >= self.y_clip_min) & (y_coords <= self.y_clip_max)

        if self.clip_by_index:
            indices = np.arange(total_points)
            visible_mask &= (indices >= self.index_clip_min) & (indices <= self.index_clip_max)

        points_to_show = self.points[visible_mask]

        # Check if the X-span is large enough to be meaningful
        if len(points_to_show) < 2:
            self.xz_axes.clear()
            self.xz_axes.text(0.5, 0.5, 'Not enough points to plot profile.',
                              horizontalalignment='center', verticalalignment='center')
            self.xz_figure.tight_layout()
            self.xz_canvas.draw()
            return

        x_span = np.ptp(points_to_show[:, 0])
        if x_span < self.xz_plot_min_x_span_m:
            self.xz_axes.clear()
            self.xz_axes.text(0.5, 0.5, f'X-span ({x_span:.2f}m) is below the minimum of {self.xz_plot_min_x_span_m}m.',
                              horizontalalignment='center', verticalalignment='center', wrap=True)
            self.xz_figure.tight_layout()
            self.xz_canvas.draw()
            return

        # Create a scaled copy for plotting, same as the 3D view
        display_points = points_to_show.copy()
        if self.z_scale_factor != 1.0 and len(display_points) > 0:
            z_mean = np.mean(display_points[:, 2])
            display_points[:, 2] = z_mean + (display_points[:, 2] - z_mean) * self.z_scale_factor

        def fit_and_plot_trendline(x_coords, z_coords, color='r', linewidth=2.0, alpha=1.0, trendline_points=500):
            """Helper to fit and plot a trendline, with optional outlier removal."""
            if len(x_coords) <= 20:
                return

            x_to_fit, z_to_fit = x_coords, z_coords

            # Outlier removal using IQR
            if self.xz_plot_remove_outliers and len(z_to_fit) > 1:
                q1 = np.percentile(z_to_fit, 25)
                q3 = np.percentile(z_to_fit, 75)
                iqr = q3 - q1
                # A factor of 0 means no outlier removal, so we check for that
                if iqr > 1e-9 and self.xz_plot_outlier_iqr_factor > 0:
                    lower_bound = q1 - (self.xz_plot_outlier_iqr_factor * iqr)
                    upper_bound = q3 + (self.xz_plot_outlier_iqr_factor * iqr)
                    inlier_mask = (z_to_fit >= lower_bound) & (z_to_fit <= upper_bound)
                    x_to_fit = x_to_fit[inlier_mask]
                    z_to_fit = z_to_fit[inlier_mask]

            if len(x_to_fit) <= 20:
                print("Not enough points left after outlier removal to fit trendline.")
                return

            # Use selected fitting method
            if self.xz_plot_fit_method == "LOWESS" and statsmodels_available:
                try:
                    # LOWESS returns smoothed points. The first column is x, second is y.
                    # It requires sorted x values for plotting.
                    smoothed = lowess(z_to_fit, x_to_fit, frac=self.xz_plot_lowess_frac)
                    x_trend = smoothed[:, 0]
                    z_trend = smoothed[:, 1]
                except Exception as e:
                    print(f"Could not fit LOWESS: {e}")
                    return
            else:  # Default to Polynomial
                try:
                    coeffs = np.polyfit(x_to_fit, z_to_fit, 20)
                    poly = np.poly1d(coeffs)  # No need to create x_trend here, it's done below
                    x_trend = np.linspace(x_coords.min(), x_coords.max(), trendline_points)  # Create x_trend for poly
                    z_trend = poly(x_trend)  # Calculate z_trend for poly
                except (np.linalg.LinAlgError, np.polynomial.polyutils.RankWarning):
                    print("Could not fit polynomial trendline. The data may be poorly conditioned.")
                    return

            # Clip the trendline to the visible plot area
            # This prevents the fit from flying out of bounds in sparse areas.
            z_trend_clipped = np.clip(z_trend, self.xz_plot_z_min, self.xz_plot_z_max)
            self.xz_axes.plot(x_trend, z_trend_clipped, '-', linewidth=linewidth, color=color, alpha=alpha)

        self.xz_axes.clear()

        # Set background colors for Z-range
        # Set the default background color for the axes patch
        self.xz_axes.set_facecolor('lightpink')
        # Add a semi-transparent yellow band for the wider Z-range [-1cm, 1cm]
        self.xz_axes.axhspan(-1, 1, facecolor='lightyellow', alpha=0.5)
        # Add a semi-transparent green band for the narrow Z-range [-0.5cm, 0.5cm] on top
        self.xz_axes.axhspan(-0.5, 0.5, facecolor='lightgreen', alpha=0.5)

        if self.xz_plot_use_sections:
            # --- Section plot based on Y-axis values ---
            plot_title = f"X-Z Profile by Y-Section ({len(display_points)} pts)"
            y_coords_all = points_to_show[:, 1]
            if len(y_coords_all) > 0:
                y_min_all, y_max_all = np.min(y_coords_all), np.max(y_coords_all)
                y_range = y_max_all - y_min_all
                num_sections = self.xz_plot_y_sections if y_range > 1e-6 else 1
                section_width = y_range / num_sections
                section_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

                for i in range(num_sections):
                    section_y_min = y_min_all + i * section_width
                    section_y_max = y_min_all + (i + 1) * section_width if i < num_sections - 1 else y_max_all + 1e-6
                    section_mask = (y_coords_all >= section_y_min) & (y_coords_all < section_y_max)
                    display_section_points = display_points[section_mask]
                    if len(display_section_points) < 2:
                        continue

                    # Also filter subsections by their X-span
                    section_x_span = np.ptp(display_section_points[:, 0])
                    if section_x_span < self.xz_plot_min_x_span_m:
                        continue  # Skip this section if its span is too small

                    # Split section into subgroups based on X-axis gaps
                    # Sort points by X to find gaps
                    sort_indices = np.argsort(display_section_points[:, 0])
                    sorted_section_points = display_section_points[sort_indices]

                    # Find gaps larger than the threshold
                    x_coords_sorted = sorted_section_points[:, 0]
                    x_diffs = np.diff(x_coords_sorted)
                    gap_indices = np.where(x_diffs > self.xz_plot_section_gap_m)[0] + 1

                    # Split the points into subgroups
                    point_subgroups = np.split(sorted_section_points, gap_indices)
                    color = section_colors[i % len(section_colors)]
                    has_labeled_section = False

                    for subgroup in point_subgroups:
                        if len(subgroup) < 2:
                            continue

                        points_to_plot_sec = subgroup
                        if len(subgroup) > 50000:
                            choice = np.random.choice(len(subgroup), 50000, replace=False)
                            points_to_plot_sec = subgroup[choice]

                        x_coords_sec = points_to_plot_sec[:, 0]
                        z_coords_sec = -points_to_plot_sec[:, 2] / self.z_scale_factor * 100

                        label = f'Y: {section_y_min:.2f}-{section_y_max:.2f}m' if not has_labeled_section else None
                        self.xz_axes.plot(x_coords_sec, z_coords_sec, '.', markersize=2, alpha=0.6, color=color,
                                          label=label)
                        fit_and_plot_trendline(x_coords_sec, z_coords_sec, color=color, linewidth=2.5, alpha=0.9,
                                               trendline_points=200)
                        has_labeled_section = True

                self.xz_axes.legend(title="Y-Sections", fontsize='small', loc='lower left')
        else:
            # --- Global Profile Plot ---
            points_to_plot = display_points
            if len(display_points) > 100000:
                choice = np.random.choice(len(display_points), 100000, replace=False)
                points_to_plot = display_points[choice]
                plot_title = f"X-Z Profile (100k of {len(display_points)})"
            else:
                plot_title = f"X-Z Profile ({len(display_points)} points)"
            x_coords = points_to_plot[:, 0]
            z_coords = -points_to_plot[:, 2] / self.z_scale_factor * 100
            self.xz_axes.plot(x_coords, z_coords, '.', markersize=2, alpha=0.7, label='All Points')
            fit_and_plot_trendline(x_coords, z_coords, color='r', linewidth=2.0)
            self.xz_axes.legend(loc='lower left')

        # Common final steps for both plot types
        self.xz_axes.set_xlabel("X-coordinate (m)")
        self.xz_axes.set_ylabel("Z-coordinate (cm)")
        self.xz_axes.set_title(plot_title)
        self.xz_axes.grid(True)
        self.xz_axes.set_aspect('auto', adjustable='box')
        self.xz_axes.set_ylim(self.xz_plot_z_min, self.xz_plot_z_max)

        # Apply fixed X-axis limits if enabled
        if self.xz_plot_fix_x_axis:
            self.xz_axes.set_xlim(self.xz_plot_x_min, self.xz_plot_x_max)

        self.xz_figure.tight_layout()
        self.xz_canvas.draw()

    def UpdateHistogramPlot(self, z_data=None):
        """Creates or updates the Z-value histogram plot based on visible data."""
        if self.hist_axes is None:
            return

        self.hist_axes.clear()

        # Set labels and title regardless of data
        self.hist_axes.set_title("Z-Height Distribution")
        self.hist_axes.set_xlabel("Z-value")
        self.hist_axes.set_ylabel("Point Count (log)")
        self.hist_axes.set_yticks([])  # Clean look

        # Set a fixed x-axis range based on the full dataset for consistent view
        if self.points is not None and len(self.points) > 0:
            self.hist_axes.set_xlim(self.z_data_min, self.z_data_max)

        # Only plot histogram if there's data to show
        if z_data is not None and len(z_data) > 0:
            # Create the histogram, getting the patches (the bars)
            n, bins, patches = self.hist_axes.hist(z_data, bins=100, log=True)

            # Color histogram bars based on Z-slider range
            colormap = get_colormap('viridis')
            # Create a normalizer that maps the selected color range to the colormap range (0-1)
            norm = matplotlib.colors.Normalize(vmin=self.z_color_min, vmax=self.z_color_max)

            for bin_center, patch in zip(bins[:-1] + np.diff(bins) / 2, patches):
                # If the bar's center is outside the selected range, color it gray
                if not (self.z_color_min <= bin_center <= self.z_color_max):
                    patch.set_facecolor('#808080')  # Gray
                else:
                    # Otherwise, apply the corresponding color from the colormap
                    # .rgb provides a (r, g, b) tuple with values in [0, 1], which is what matplotlib expects.
                    patch.set_facecolor(colormap[norm(bin_center)].rgb)

        # Add vertical lines for min/max controls, drawn on top
        self.hist_min_line = self.hist_axes.axvline(self.z_color_min, color='cyan', linestyle='--')
        self.hist_max_line = self.hist_axes.axvline(self.z_color_max, color='magenta', linestyle='--')

        self.hist_figure.tight_layout(pad=0.5)
        self.hist_canvas.draw()

    def OnCalculateNormals(self, event):
        if self.points is None:
            return
        self.SetStatusText("Calculating normals...")
        wx.BeginBusyCursor()
        threading.Thread(target=self.CalculateNormalsThread).start()

    def CalculateNormalsThread(self):
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            extent = np.max(np.ptp(self.points, axis=0))
            radius = extent * 0.05
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)
            normals = np.asarray(pcd.normals)
            wx.CallAfter(self.OnNormalsComplete, normals)
        except Exception as e:
            print(f"Error calculating normals: {e}")
            wx.CallAfter(self.OnNormalsComplete, None)

    def OnNormalsComplete(self, normals):
        wx.EndBusyCursor()
        if normals is None:
            wx.LogError("Failed to calculate normals.")
            self.SetStatusText(f"Loaded {len(self.points)} points | Normal calculation failed.")
            return

        self.normals = normals
        self.SetStatusText(f"Loaded {len(self.points)} points | Normals calculated.")
        self.show_normals_item.Enable(True)
        self.show_normals_item.Check(True)
        self.show_normals = True
        self.UpdateDisplay()

    def _clear_preloader(self):
        """Wait for any existing preloading threads to finish."""
        [t.join() for t in self.preloading_threads if t.is_alive()]

    def OnShowNormals(self, event):
        self.show_normals = event.IsChecked()
        self.UpdateDisplay()

    def StartLoadingFile(self, filepath):
        try:
            current_dir = os.path.dirname(filepath)
            filename = os.path.basename(filepath)

            compatible_files = sorted([
                f for f in os.listdir(current_dir)
                if f.lower().endswith(('.pcd', '.ply'))
            ])

            if filename in compatible_files:
                self.current_dir = current_dir
                self.file_list = compatible_files
                self.current_file_index = self.file_list.index(filename)
            else:
                self.file_list = [filename]
                self.current_dir = current_dir
                self.current_file_index = 0
        except Exception as e:
            print(f"Could not establish file list for navigation: {e}")
            self.file_list = []
            self.current_file_index = -1
            self.current_dir = None

        self.SetStatusText(f"Opening {os.path.basename(filepath)}...")
        # If loading on-demand, check cache first.
        with self.cache_lock:
            cached_data = self.preload_cache.pop(filepath, None)

        if cached_data:
            print(f"Loading {os.path.basename(filepath)} from cache.")
            self.OnLoadComplete(*cached_data)
            return

        wx.BeginBusyCursor()
        threading.Thread(target=self.LoadThread, args=(filepath,)).start()

    def LoadThread(self, filepath):
        """Worker thread for on-demand file loading."""
        loaded_data = self._load_file_data(filepath)
        wx.CallAfter(self.OnLoadComplete, *loaded_data)

    def _load_associated_images(self, pcd_filepath):
        """Helper to load .jpg and _p.jpg images for a given point cloud file."""
        image_bitmap = None
        projected_image_bitmap = None
        try:
            base_name = os.path.splitext(os.path.basename(pcd_filepath))[0]
            pcd_dir = os.path.dirname(pcd_filepath)
            parent_dir = os.path.dirname(pcd_dir)
            image_path = os.path.join(parent_dir, 'camera', 'front', base_name + '.jpg')
            projected_image_path = os.path.join(parent_dir, 'camera', 'front', base_name + '_p.jpg')

            if os.path.exists(image_path):
                wx_image = wx.Image(image_path, wx.BITMAP_TYPE_ANY)
                if wx_image.IsOk():
                    image_bitmap = wx_image.ConvertToBitmap()

            if os.path.exists(projected_image_path):
                wx_proj_image = wx.Image(projected_image_path, wx.BITMAP_TYPE_ANY)
                if wx_proj_image.IsOk():
                    projected_image_bitmap = wx_proj_image.ConvertToBitmap()

        except Exception as e:
            print(f"Error loading associated images for {pcd_filepath}: {e}")

        return image_bitmap, projected_image_bitmap

    def _load_file_data(self, filepath):
        """Loads all data for a given filepath. This is the core loading logic."""
        points, colors = open_point_cloud_file(filepath)
        image_bitmap, projected_image_bitmap = self._load_associated_images(filepath)
        return points, colors, image_bitmap, os.path.basename(filepath), projected_image_bitmap

    def _preload_worker(self, filepath_to_load):
        """Worker thread for preloading a single file into the cache."""
        with self.cache_lock:
            if filepath_to_load in self.preload_cache:
                return  # Already cached or being cached

        # Load data outside the lock
        loaded_data = self._load_file_data(filepath_to_load)

        with self.cache_lock:
            self.preload_cache[filepath_to_load] = loaded_data
            print(f"Preloaded {os.path.basename(filepath_to_load)} into cache.")

    def OnLoadComplete(self, points, colors, image_bitmap, filename, projected_image_bitmap):
        if wx.IsBusy():
            wx.EndBusyCursor()

        if points is None:
            wx.LogError(f"Failed to open or read point cloud from file: {filename}.")
            self.SetStatusText("Failed to load file.")
            return

        self.points = points
        self.colors = colors
        self.normals = None
        self.corresponding_image = image_bitmap
        self.projected_image = projected_image_bitmap

        nav_info = ""
        if self.file_list:
            nav_info = f"({self.current_file_index + 1}/{len(self.file_list)})"
        self.SetTitle(f"Point Cloud Viewer: {filename} {nav_info} | InViLab - UAntwerpen")

        # Make view settings persistent across file loads
        has_colors = self.colors is not None
        self.toggle_colors_item.Enable(has_colors)
        if not has_colors:
            # If the new file has no colors, we must disable the color-showing mode.
            self.show_colors = False

        # Update menu items to reflect the persistent state of the view options.
        self.toggle_colors_item.Check(self.show_colors)
        self.flip_z_item.Check(self.flip_z)
        self.show_grid_item.Check(self.show_grid)
        self.color_by_z_item.Check(self.color_by_z)
        self.clip_by_y_item.Check(self.clip_by_y)
        self.clip_by_index_item.Check(self.clip_by_index)

        # Enable controls that are always available for a loaded cloud.
        self.flip_z_item.Enable(True)
        self.color_by_z_item.Enable(True)
        self.clip_by_y_item.Enable(True)
        self.clip_by_index_item.Enable(True)

        # Default Z-coloring range to the 5th-95th percentile
        z_coords = self.points[:, 2]
        self.z_data_min = np.min(z_coords)
        self.z_data_max = np.max(z_coords)
        z_data_range = self.z_data_max - self.z_data_min

        # Default to the 5%-95% interval to ignore outliers and improve contrast
        self.z_color_min = np.percentile(z_coords, 5)
        self.z_color_max = np.percentile(z_coords, 95)

        # Update slider positions to match the new default range
        min_z_slider_pos = int(
            ((self.z_color_min - self.z_data_min) / z_data_range) * 1000) if z_data_range > 1e-6 else 0
        max_z_slider_pos = int(
            ((self.z_color_max - self.z_data_min) / z_data_range) * 1000) if z_data_range > 1e-6 else 1000

        self.z_min_slider.SetValue(min_z_slider_pos)
        self.z_max_slider.SetValue(max_z_slider_pos)

        is_z_color_enabled = self.color_by_z and self.points is not None
        self.z_min_slider.Enable(is_z_color_enabled)
        self.z_max_slider.Enable(is_z_color_enabled)
        self.hist_canvas.Enable(is_z_color_enabled)

        # Initialize Y-clipping state on load
        y_coords = self.points[:, 1]
        self.y_data_min = np.min(y_coords)
        self.y_data_max = np.max(y_coords)
        self.y_clip_min = self.y_data_min
        self.y_clip_max = self.y_data_max
        self.y_min_slider.SetValue(0)
        self.y_max_slider.SetValue(1000)
        is_y_clip_enabled = self.clip_by_y and self.points is not None
        self.y_min_slider.Enable(is_y_clip_enabled)
        self.y_max_slider.Enable(is_y_clip_enabled)

        # Initialize Index-clipping state on load
        self.index_data_min = 0
        self.index_data_max = len(self.points) - 1
        self.index_clip_min = self.index_data_min
        self.index_clip_max = self.index_data_max
        self.index_min_slider.SetValue(0)
        self.index_max_slider.SetValue(1000)
        is_index_clip_enabled = self.clip_by_index and self.points is not None
        self.index_min_slider.Enable(is_index_clip_enabled)
        self.index_max_slider.Enable(is_index_clip_enabled)

        # Normals are not persistent as they must be recalculated for each file.
        self.calculate_normals_item.Enable(True)
        self.show_normals_item.Enable(False)
        self.show_normals_item.Check(False)
        self.show_normals = False

        self.point_size_slider.Enable(True)

        # Enable point style controls
        self.symbol_choice.Enable(True)
        self.symbol_choice.SetStringSelection(self.point_symbol)

        # Enable and reset the Z-Scale slider on new file load
        self.z_scale_slider.Enable(True)
        self.z_scale_factor = 1.0
        self.z_scale_slider.SetValue(10)  # Reset to 1.0x scale

        # Enable the XZ plot canvas and its controls
        self.xz_canvas.Enable(True)
        self.xz_plot_sections_button.Enable(True)
        self.xz_remove_outliers_button.Enable(True)
        self.fit_method_choice.Enable(True)
        self.xz_z_min_ctrl.Enable(True)
        self.xz_z_max_ctrl.Enable(True)
        self.xz_z_min_ctrl.SetValue(f"{self.xz_plot_z_min:.2f}")
        self.xz_z_max_ctrl.SetValue(f"{self.xz_plot_z_max:.2f}")
        self.xz_fix_x_axis_button.Enable(True)
        self.xz_x_min_ctrl.Enable(self.xz_plot_fix_x_axis)
        self.xz_x_max_ctrl.Enable(self.xz_plot_fix_x_axis)
        self.xz_x_min_ctrl.SetValue(f"{self.xz_plot_x_min:.2f}")
        self.xz_x_max_ctrl.SetValue(f"{self.xz_plot_x_max:.2f}")

        # Enable/disable axis control buttons
        self.xz_autoscale_button.Enable(True)
        self.xz_defaults_button.Enable(True)
        self.xz_previous_settings_available = False  # Reset on new file
        self.xz_previous_button.Enable(False)

        # Update projection button state
        can_project = self.projected_image is not None
        self.projection_button.Enable(can_project)
        if not can_project and self.show_projection:
            # If we can no longer project, turn it off
            self.show_projection = False
            self.projection_button.SetValue(False)

        # Update label button states based on CSV file
        self.UpdateLabelButtonStates()

        # --- Trigger Preloading and Cache Management ---
        self._clear_preloader()  # Ensure previous preloading is done
        self.preloading_threads.clear()

        if self.file_list and len(self.file_list) > 1:
            # 1. Manage cache size
            with self.cache_lock:
                # Determine which indices to keep in the cache
                current_idx = self.current_file_index
                cache_radius = self.CACHE_SIZE // 2
                indices_to_keep = {
                    (current_idx + i) % len(self.file_list)
                    for i in range(-cache_radius, cache_radius + 1)
                }
                filepaths_to_keep = {os.path.join(self.current_dir, self.file_list[i]) for i in indices_to_keep}

                # Remove keys that are no longer nearby
                keys_to_remove = [key for key in self.preload_cache if key not in filepaths_to_keep]
                for key in keys_to_remove:
                    del self.preload_cache[key]
                    print(f"Removed {os.path.basename(key)} from cache.")

            # 2. Start preloading neighbors
            indices_to_preload = set()
            # Preload 5 files forward and 5 files backward
            for i in range(1, 6):
                indices_to_preload.add((self.current_file_index + i) % len(self.file_list))
                indices_to_preload.add((self.current_file_index - i) % len(self.file_list))

            for idx in indices_to_preload:
                filepath = os.path.join(self.current_dir, self.file_list[idx])
                thread = threading.Thread(target=self._preload_worker, args=(filepath,))
                self.preloading_threads.append(thread)
                thread.start()

        self.UpdateDisplay()
        self.OnResetView()

        # Refresh the image panel
        self.image_panel.Refresh()

        # Set focus to the canvas after loading to ensure keys are captured
        self.canvas.native.SetFocus()

    def OnExit(self, event):
        self.Close()


if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))

    app = wx.App(False)
    filepath_on_open = sys.argv[1] if len(sys.argv) > 1 else None
    frame = PCDViewerFrame(None, filepath=filepath_on_open, size=(1280, 960))
    app.MainLoop()