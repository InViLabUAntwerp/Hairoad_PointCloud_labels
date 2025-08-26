#
# copyright Seppe Sels 2025
#
# This code is for internal use only (Uantwerpen, project members)
# Bugs, bugfixes and additions to the code need to be reported to Invilab (contact: Seppe Sels)
# GUI: AI generated
# --- MODIFIED FOR PCD/PLY FILE VISUALIZATION WITH NORMALS & CUSTOM COLORS ---
#

import wx
import numpy as np
import os
import sys
import threading
import open3d as o3d
import json
import csv

# --- ADDED: Imports for 3D Visualization ---
from vispy import scene
from vispy.scene import cameras
from vispy.scene.visuals import Text, Line
from vispy.color import Color, get_colormap

# Add PIL import for image loading
from PIL import Image

# --- NEW: Add Matplotlib for histogram plotting ---
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


class XZPlotFrame(wx.Frame):
    """A separate frame to display the X-Z scatter plot."""

    def __init__(self, parent, title="X-Z Profile"):
        super(XZPlotFrame, self).__init__(parent, title=title, size=(800, 600))
        self.panel = wx.Panel(self)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.panel, -1, self.figure)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.panel.SetSizer(sizer)

        self.Centre()

    def update_plot(self, points, z_scale_factor):
        if points is None or len(points) == 0:
            self.axes.clear()
            self.axes.text(0.5, 0.5, 'No points to display.', horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        # Subsample if too many points to keep the plot responsive
        if len(points) > 100000:
            choice = np.random.choice(len(points), 100000, replace=False)
            points_to_plot = points[choice]
            plot_title = f"X-Z Profile (showing 100k of {len(points)} points)"
        else:
            points_to_plot = points
            plot_title = f"X-Z Profile ({len(points)} points )"

        x_coords = points_to_plot[:, 0]
        z_coords = -points_to_plot[:, 2] / z_scale_factor * 100

        self.axes.clear()
        # Use plot with markers for performance
        self.axes.plot(x_coords, z_coords, '.', markersize=2, alpha=0.7, label='Points')

        # --- NEW: Calculate and plot 4th order polynomial trendline ---
        # Ensure there are enough points to fit a 4th order polynomial
        if len(x_coords) > 4:
            try:
                # Fit the polynomial
                coeffs = np.polyfit(x_coords, z_coords, 20)
                poly = np.poly1d(coeffs)

                # Generate smooth x-values for the trendline
                x_trend = np.linspace(x_coords.min(), x_coords.max(), 500)
                z_trend = poly(x_trend)

                # Plot the trendline
                self.axes.plot(x_trend, z_trend, 'r-', linewidth=2, label='7th Order Trendline')
                # self.axes.legend()  # Display the legend
            except np.linalg.LinAlgError:
                print("Could not fit polynomial trendline. The data may be poorly conditioned.")

        self.axes.set_xlabel("X-coordinate (m)")
        self.axes.set_ylabel(f"Z-coordinate (cm)")
        self.axes.set_title(plot_title)
        self.axes.grid(True)
        self.axes.set_aspect('auto', adjustable='box')
        self.figure.tight_layout()
        self.canvas.draw()


class PCDViewerFrame(wx.Frame):
    def __init__(self, *args, filepath=None, **kw):
        super(PCDViewerFrame, self).__init__(*args, **kw)

        self.points = None
        self.colors = None
        self.normals = None
        self.show_normals = False
        self.show_colors = False
        self.flip_z = False
        self.color_by_z = False
        self.default_point_color = 'white'

        # --- NEW: Y-clipping attributes ---
        self.clip_by_y = True
        self.y_data_min = 0.0
        self.y_data_max = 1.0
        self.y_clip_min = 0.0
        self.y_clip_max = 1.0
        self.y_min_slider = None
        self.y_max_slider = None

        # --- NEW: Z-coloring range attributes ---
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

        # --- NEW: Z-Scale attributes ---
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

        # --- NEW: XZ Plot attributes ---
        self.show_xz_plot_button = None
        self.xz_plot_frame = None

        # --- MODIFIED: Labeling attributes ---
        self.labels = []
        self.label_buttons = {}

        # --- NEW: Image display attributes ---
        self.image_panel = None
        self.corresponding_image = None  # This will be a wx.Bitmap

        self.InitUI()
        self.Maximize()
        self.LoadSettings()
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
                    # Ensure labels is a list of strings
                    if 'labels' in settings and isinstance(settings['labels'], list):
                        self.labels = [str(label) for label in settings['labels']]
                        if not self.labels:  # If list is empty
                            self.labels = default_labels
                            print("Warning: 'labels' in settings.json is empty. Using default labels.")
                    else:
                        self.labels = default_labels
                        print("Warning: 'labels' key not found or not a list in settings.json. Using default labels.")
            else:
                self.labels = default_labels
        except (json.JSONDecodeError, Exception) as e:
            self.labels = default_labels
            print(f"Error loading settings.json: {e}. Using default labels.")

    def InitUI(self):
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

        self.canvas = scene.SceneCanvas(parent=self.panel, keys='interactive', show=True, bgcolor='lightpink')
        self.view = self.canvas.central_widget.add_view()
        # --- MODIFIED: Switched to TurntableCamera for more stable rotation and clipping ---
        self.view.camera = cameras.ArcballCamera(fov=45, distance=10, up='+z')

        # --- FIX: Bind key handler directly to the VisPy canvas widget ---
        self.canvas.native.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

        scene.visuals.XYZAxis(parent=self.view.scene)
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.normals_visual = Line(parent=self.view.scene, method='gl', width=1)
        self.normals_visual.visible = False

        controls_text = (
            "Controls:\n"
            "LMB: Orbit\n"
            "RMB / Scroll: Zoom\n"
            "Shift + LMB: Pan\n"
            "Shift + RMB: FOV\n"
            "'n' / 'p': Next/Prev File"
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

        # --- MODIFIED: Major UI layout change ---
        # Left side is the 3D canvas, right side is a new controls panel
        main_sizer.Add(self.canvas.native, 3, wx.EXPAND)

        controls_panel = wx.Panel(self.panel)
        controls_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- NEW: Image Display Panel ---
        self.image_panel = wx.Panel(controls_panel, style=wx.BORDER_SUNKEN)
        self.image_panel.SetBackgroundColour(wx.BLACK)
        self.image_panel.SetMinSize((-1, 200))  # Give it some initial size
        controls_sizer.Add(self.image_panel, 1, wx.EXPAND | wx.ALL, 5)

        # --- NEW: Histogram Plot ---
        self.hist_figure = Figure(figsize=(4, 3), dpi=100)
        self.hist_axes = self.hist_figure.add_subplot(111)
        self.hist_canvas = FigureCanvas(controls_panel, -1, self.hist_figure)
        self.hist_canvas.Enable(False)
        controls_sizer.Add(self.hist_canvas, 1, wx.EXPAND | wx.ALL, 5)

        # Point Size Slider
        ps_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ps_sizer.Add(wx.StaticText(controls_panel, label="Point Size:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.point_size_slider = wx.Slider(controls_panel, value=3, minValue=1, maxValue=20)
        self.point_size_slider.Enable(False)
        ps_sizer.Add(self.point_size_slider, 1, wx.EXPAND)
        controls_sizer.Add(ps_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # --- NEW: Z-Min/Max Sliders ---
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

        # --- NEW: Z-Scale Slider ---
        z_scale_sizer = wx.BoxSizer(wx.HORIZONTAL)
        z_scale_sizer.Add(wx.StaticText(controls_panel, label="Z-Scale:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        # Slider from 10 to 200, representing 1.0x to 20.0x scale
        self.z_scale_slider = wx.Slider(controls_panel, value=10, minValue=10, maxValue=200)
        self.z_scale_slider.Enable(False)
        z_scale_sizer.Add(self.z_scale_slider, 1, wx.EXPAND)
        controls_sizer.Add(z_scale_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # --- NEW: Y-Clip Min/Max Sliders ---
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

        # --- NEW: Index Clip Min/Max Sliders ---
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

        # --- NEW: X-Z Profile Plot Button ---
        self.show_xz_plot_button = wx.Button(controls_panel, label="Show X-Z Profile")
        self.show_xz_plot_button.Enable(False)
        controls_sizer.Add(self.show_xz_plot_button, 0, wx.EXPAND | wx.ALL, 5)

        # --- MODIFIED: Labeling Buttons ---
        label_box = wx.StaticBox(controls_panel, label="Label Data")
        # Use a grid sizer for flexible layout
        self.LoadSettings()
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
        self.Bind(wx.EVT_BUTTON, self.OnShowXZPlot, self.show_xz_plot_button)

        # --- MODIFIED: Bind Labeling Buttons ---
        for button in self.label_buttons.values():
            self.Bind(wx.EVT_TOGGLEBUTTON, self.OnLabelToggle, button)
        self.Bind(wx.EVT_MENU, self.OnCalculateNormals, self.calculate_normals_item)
        self.Bind(wx.EVT_MENU, self.OnShowNormals, self.show_normals_item)
        self.Bind(wx.EVT_MENU, self.OnToggleBackground, self.dark_bg_item)
        self.Bind(wx.EVT_MENU, self.OnSetPointColor, set_color_item)
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
        else:
            event.Skip()
            return

        if new_index != self.current_file_index:
            self.current_file_index = new_index
            new_filename = self.file_list[self.current_file_index]
            new_filepath = os.path.join(self.current_dir, new_filename)
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
            self.StartLoadingFile(fd.GetPath())

    def OnPointSizeChange(self, event):
        self.UpdateDisplay()
        if self.canvas:
            self.canvas.native.SetFocus()

    def OnResetView(self, event=None):
        if self.view and self.points is not None and len(self.points) > 0:
            # --- MODIFIED: Use a scaled copy for calculating camera position ---
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

            # --- NEW: Manually set clipping planes for better rendering ---
            # This prevents points from disappearing when rotating the camera.
            self.view.camera.near = distance / 1000.0
            self.view.camera.far = distance + max_extent * 2.0
        elif self.view:
            self.view.camera.reset()
            self.view.camera.distance = 10

    def OnToggleColors(self, event):
        self.show_colors = event.IsChecked()
        # --- NEW: Ensure color modes are mutually exclusive ---
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

        # --- NEW: Ensure color modes are mutually exclusive ---
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

    def OnShowXZPlot(self, event):
        """Creates or updates the X-Z profile plot in a separate window."""
        if self.points is None:
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
        display_points = points_to_show.copy()

        if self.z_scale_factor != 1.0 and len(display_points) > 0:
            z_mean = np.mean(display_points[:, 2])
            display_points[:, 2] = z_mean + (display_points[:, 2] - z_mean) * self.z_scale_factor

        # If the frame exists, destroy it to ensure a fresh start
        if self.xz_plot_frame:
            self.xz_plot_frame.Destroy()

        self.xz_plot_frame = XZPlotFrame(self)
        self.xz_plot_frame.update_plot(display_points, self.z_scale_factor)
        self.xz_plot_frame.Show()

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

        if self.corresponding_image and self.corresponding_image.IsOk():
            img_w, img_h = self.corresponding_image.GetWidth(), self.corresponding_image.GetHeight()

            if panel_w == 0 or panel_h == 0 or img_w == 0 or img_h == 0:
                return

            # Calculate scale to fit image while maintaining aspect ratio
            scale = min(panel_w / img_w, panel_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)

            # Center the image
            x = (panel_w - new_w) // 2
            y = (panel_h - new_h) // 2

            # Create a scaled bitmap
            image = self.corresponding_image.ConvertToImage()
            image.Rescale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)
            scaled_bitmap = wx.Bitmap(image)

            dc.DrawBitmap(scaled_bitmap, x, y, useMask=False)

    def OnSizeImage(self, event):
        if self.image_panel:
            self.image_panel.Refresh()
        event.Skip()

    def UpdateDisplay(self):
        if self.points is None:
            self.markers.set_data(np.empty((0, 3)))
            self.SetStatusText("No file loaded.")
            return

        point_size = self.point_size_slider.GetValue()

        # --- REFACTORED: Use a single boolean mask for all filters ---
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

        # --- NEW: Update the histogram with the Z-data of the *visible* points ---
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

        self.markers.set_data(display_points, size=point_size, edge_color=None, face_color=display_colors)

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

        # --- REFACTORED: Update normals visual based on visible and scaled points ---
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

        self.canvas.update()

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

            # --- NEW: Color histogram bars based on Z-slider range ---
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
        wx.BeginBusyCursor()
        threading.Thread(target=self.LoadThread, args=(filepath,)).start()

    def LoadThread(self, filepath):
        points, colors = open_point_cloud_file(filepath)

        # --- NEW: Load corresponding image ---
        image_bitmap = None
        try:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            pcd_dir = os.path.dirname(filepath)
            parent_dir = os.path.dirname(pcd_dir)
            # The path is one dir up, then /camera/front/*.jpg
            image_path = os.path.join(parent_dir, 'camera', 'front', base_name + '.jpg')

            if os.path.exists(image_path):
                # Use wx.Image to load, it supports more formats like jpg
                wx_image = wx.Image(image_path, wx.BITMAP_TYPE_ANY)
                if wx_image.IsOk():
                    image_bitmap = wx_image.ConvertToBitmap()
                    print(f"Found and loaded corresponding image: {image_path}")
                else:
                    print(f"Found image file, but failed to load: {image_path}")
            else:
                print(f"Corresponding image not found at: {image_path}")
        except Exception as e:
            print(f"Error while searching for or loading corresponding image: {e}")

        wx.CallAfter(self.OnLoadComplete, points, colors, image_bitmap, os.path.basename(filepath))

    def OnLoadComplete(self, points, colors, image_bitmap, filename):
        wx.EndBusyCursor()
        if points is None:
            wx.LogError(f"Failed to open or read point cloud from file: {filename}.")
            self.SetStatusText("Failed to load file.")
            return

        self.points = points
        self.colors = colors
        self.normals = None
        self.corresponding_image = image_bitmap

        nav_info = ""
        if self.file_list:
            nav_info = f"({self.current_file_index + 1}/{len(self.file_list)})"
        self.SetTitle(f"Point Cloud Viewer: {filename} {nav_info} | InViLab - UAntwerpen")

        # --- MODIFIED: Make view settings persistent across file loads ---
        has_colors = self.colors is not None
        self.toggle_colors_item.Enable(has_colors)
        if not has_colors:
            # If the new file has no colors, we must disable the color-showing mode.
            self.show_colors = False

        # Update menu items to reflect the persistent state of the view options.
        self.toggle_colors_item.Check(self.show_colors)
        self.flip_z_item.Check(self.flip_z)
        self.color_by_z_item.Check(self.color_by_z)
        self.clip_by_y_item.Check(self.clip_by_y)
        self.clip_by_index_item.Check(self.clip_by_index)

        # Enable controls that are always available for a loaded cloud.
        self.flip_z_item.Enable(True)
        self.color_by_z_item.Enable(True)
        self.clip_by_y_item.Enable(True)
        self.clip_by_index_item.Enable(True)

        # --- MODIFIED: Default Z-coloring range to the 5th-95th percentile ---
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

        # --- NEW: Initialize Y-clipping state on load ---
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

        # --- NEW: Initialize Index-clipping state on load ---
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

        # --- FIX: Enable and reset the Z-Scale slider on new file load ---
        self.z_scale_slider.Enable(True)
        self.z_scale_factor = 1.0
        self.z_scale_slider.SetValue(10)  # Reset to 1.0x scale

        # --- NEW: Enable the XZ plot button ---
        self.show_xz_plot_button.Enable(True)

        # --- NEW: Update label button states based on CSV file ---
        self.UpdateLabelButtonStates()

        self.UpdateDisplay()
        self.OnResetView()

        # --- NEW: Refresh the image panel ---
        self.image_panel.Refresh()

        # --- FIX: Set focus to the canvas after loading to ensure keys are captured ---
        self.canvas.native.SetFocus()

    def OnExit(self, event):
        if self.xz_plot_frame:
            self.xz_plot_frame.Close()
        self.Close()


if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))

    app = wx.App(False)
    filepath_on_open = sys.argv[1] if len(sys.argv) > 1 else None
    frame = PCDViewerFrame(None, filepath=filepath_on_open, size=(1280, 960))
    app.MainLoop()