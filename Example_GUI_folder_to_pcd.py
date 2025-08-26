#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wx
import wx.adv  # For TaskBarIcon
import os
import glob
import threading
import sys
import json  # For saving/loading config
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # <-- SET THE BACKEND HERE
import matplotlib.pyplot as plt
from timesync.timesync_crawler import process_master_file

# --- Constants ---
CONFIG_FILE = "config.json"
APP_NAME = "Hairoad Data Processor"
ICON_FILE = "Invilab_ico.ico"  # Icon for the frame and system tray

# --- Custom Events for Thread-Safe UI Updates ---
myEVT_LOG_TYPE = wx.NewEventType()
EVT_LOG = wx.PyEventBinder(myEVT_LOG_TYPE, 1)

myEVT_STATUS_TYPE = wx.NewEventType()
EVT_STATUS = wx.PyEventBinder(myEVT_STATUS_TYPE, 1)

myEVT_FINISH_TYPE = wx.NewEventType()
EVT_FINISH = wx.PyEventBinder(myEVT_FINISH_TYPE, 1)


class CustomEvent(wx.PyEvent):
    """Simple event to carry data."""

    def __init__(self, event_type, data):
        super().__init__()
        self.SetEventType(event_type)
        self.data = data


# --- Helper Functions ---
def resource_path(relative_path):
    """ Get absolute path to a resource, works for dev and for cx_Freeze/PyInstaller. """
    if getattr(sys, 'frozen', False):
        # The application is frozen
        base_path = os.path.dirname(sys.executable)
    else:
        # The application is running in a normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# --- Stdout/Stderr Redirector ---
class WxTextCtrlRedirector:
    """Redirects print statements to a wx.TextCtrl via thread-safe events."""

    def __init__(self, target_window):
        self.target_window = target_window

    def write(self, string):
        # Post a log event instead of using CallAfter for consistency
        wx.PostEvent(self.target_window, CustomEvent(myEVT_LOG_TYPE, string))

    def flush(self):
        pass  # Not needed


# --- Worker Thread ---
class ProcessingThread(threading.Thread):
    def __init__(self, parent_window, config, stop_event):
        super().__init__()
        self._parent = parent_window
        self._config = config
        self._stop_event = stop_event
        self.was_stopped = False

    def run(self):
        # Redirect stdout/stderr to the GUI log
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = WxTextCtrlRedirector(self._parent)

        try:
            self._process_files()
        except Exception as e:
            import traceback
            print(f"\n\n❌ An unhandled error occurred in the worker thread: {e}")
            print(traceback.format_exc())
            self._post_status(f"Error: {e}")
        finally:
            # Restore stdout/stderr and signal completion
            sys.stdout, sys.stderr = original_stdout, original_stderr
            wx.PostEvent(self._parent, CustomEvent(myEVT_FINISH_TYPE, None))

    def _post_status(self, message):
        """Helper to post a status update event."""
        wx.PostEvent(self._parent, CustomEvent(myEVT_STATUS_TYPE, message))

    def _process_files(self):
        """Main processing logic, imported from your library."""
        # This import should be inside the function that uses it if it's a heavy one
        # or has dependencies that are only needed here.

        master_dir = self._config['master_dir']
        output_dir = self._config['output_dir']

        if not os.path.exists(output_dir):
            print(f"Output directory does not exist. Creating: {output_dir}")
            os.makedirs(output_dir)

        self._post_status("Searching for master files...")
        master_file_list = glob.glob(os.path.join(master_dir, "*.bin.txt"))
        total_files = len(master_file_list)

        if not master_file_list:
            print(f"❌ No master files (*.bin.txt) found in '{master_dir}'.")
            self._post_status("No files found. Check Master Directory.")
            return

        print(f"Found {total_files} master files to process.")
        for i, master_file in enumerate(master_file_list):
            if self._stop_event.is_set():
                print("\n🛑 Processing stopped by user.")
                self.was_stopped = True
                self._post_status("Stopped by user.")
                break

            status_msg = f"Processing file {i + 1}/{total_files}: {os.path.basename(master_file)}"
            self._post_status(status_msg)
            print(f"\n--- {status_msg} ---")

            process_master_file(
                master_file,
                self._config['camera_dir'],
                output_dir,
                border=self._config['border'],
                TIME_DIFF_THRESHOLD=self._config['time_diff_threshold'],
                CALIBRATION_FILE = 'calib1.h5',
                TRANSFORMATION_FILE = 'transformation_and_intrinsics7.txt',
            )

        if not self.was_stopped:
            print("\n\n✅ All files processed successfully.")
            self._post_status("Processing complete.")


# --- System Tray Icon ---
class AppTaskBarIcon(wx.adv.TaskBarIcon):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame

        # Set the icon
        icon_path = resource_path(ICON_FILE)
        if os.path.exists(icon_path):
            icon = wx.Icon(icon_path, wx.BITMAP_TYPE_ICO)
            self.SetIcon(icon, APP_NAME)

        # Bind events
        self.Bind(wx.adv.EVT_TASKBAR_LEFT_DCLICK, self.on_show_frame)
        self.Bind(wx.EVT_MENU, self.on_show_frame, id=1)
        self.Bind(wx.EVT_MENU, self.on_exit_app, id=2)

    def CreatePopupMenu(self):
        menu = wx.Menu()
        menu.Append(1, "Show Processor")
        menu.AppendSeparator()
        menu.Append(2, "Exit")
        return menu

    def on_show_frame(self, event):
        self.frame.Show()
        self.frame.Restore()
        self.frame.Raise()

    def on_exit_app(self, event):
        self.frame.Close()


# --- Main GUI Frame ---
class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=APP_NAME, size=(750, 650))
        self.config = self._load_config()
        self.worker = None
        self.stop_event = None

        self._init_ui()
        self._bind_events()

        # Set frame and taskbar icon
        self.taskBarIcon = AppTaskBarIcon(self)
        icon_path = resource_path(ICON_FILE)
        if os.path.exists(icon_path):
            self.SetIcon(wx.Icon(icon_path, wx.BITMAP_TYPE_ICO))

        self.Centre()
        self.Show()

    def _init_ui(self):
        """Creates and lays out all the widgets."""
        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Input Fields
        input_sizer = self._create_input_panel()
        main_sizer.Add(input_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Control Buttons
        controls_sizer = self._create_controls_panel()
        main_sizer.Add(controls_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        # Status Label
        self.status_label = wx.StaticText(self.panel, label="Status: Idle")
        main_sizer.Add(self.status_label, 0, wx.ALIGN_CENTER | wx.BOTTOM, 5)

        # Log Window
        log_sizer = self._create_log_panel()
        main_sizer.Add(log_sizer, 1, wx.EXPAND | wx.ALL, 10)

        self.panel.SetSizer(main_sizer)

    def _create_input_panel(self):
        """Creates the grid of input text fields and browse buttons."""
        sizer = wx.FlexGridSizer(cols=3, gap=(5, 10))
        sizer.AddGrowableCol(1)

        self.master_dir_txt = self._create_path_row(sizer, "Master Files Dir:", 'master_dir')
        self.camera_dir_txt = self._create_path_row(sizer, "Camera Data Dir:", 'camera_dir')
        self.output_dir_txt = self._create_path_row(sizer, "Output Dir:", 'output_dir')
        self.border_txt = self._create_text_row(sizer, "Border Size:", 'border')
        self.time_diff_txt = self._create_text_row(sizer, "Time Diff Threshold (s):", 'time_diff_threshold')

        return sizer

    def _create_controls_panel(self):
        """Creates the Start and Stop buttons."""
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.start_button = wx.Button(self.panel, label="Start Processing")
        self.stop_button = wx.Button(self.panel, label="Stop")
        self.stop_button.Disable()
        sizer.Add(self.start_button, 0, wx.ALL, 5)
        sizer.Add(self.stop_button, 0, wx.ALL, 5)
        return sizer

    def _create_log_panel(self):
        """Creates the log output label and text control."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        log_label = wx.StaticText(self.panel, label="Log Output:")
        self.log_txt = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        sizer.Add(log_label, 0, wx.BOTTOM, 5)
        sizer.Add(self.log_txt, 1, wx.EXPAND)
        return sizer

    def _bind_events(self):
        """Binds all event handlers."""
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)
        self.Bind(EVT_LOG, self.on_log_result)
        self.Bind(EVT_STATUS, self.on_status_update)
        self.Bind(EVT_FINISH, self.on_finish)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def on_start(self, event):
        """Starts the processing thread."""
        if not self._update_and_validate_config():
            return

        self.log_txt.Clear()
        self.log_txt.AppendText(f"[{datetime.now().strftime('%H:%M:%S')}] Starting process...\n")
        self.status_label.SetLabel("Status: Initializing...")

        self._toggle_controls(is_running=True)
        self.stop_event = threading.Event()
        self.worker = ProcessingThread(self, self.config, self.stop_event)
        self.worker.start()

    def on_stop(self, event):
        """Signals the worker thread to stop."""
        if self.worker and self.stop_event:
            self.status_label.SetLabel("Status: Sending stop signal...")
            self.log_txt.AppendText("\n--- Sending stop signal... ---\n")
            self.stop_event.set()
            self.stop_button.Disable()

    def on_log_result(self, event):
        """Appends log messages from the worker thread to the text control."""
        # This logic correctly handles overwriting progress lines that use carriage return
        text = event.data
        if text.startswith('\r'):
            last_line_start = self.log_txt.XYToPosition(0, self.log_txt.GetNumberOfLines() - 1)
            self.log_txt.Replace(last_line_start, self.log_txt.GetLastPosition(), text.lstrip('\r'))
        else:
            self.log_txt.AppendText(text)

    def on_status_update(self, event):
        """Updates the status label from the worker thread."""
        self.status_label.SetLabel(f"Status: {event.data}")

    def on_finish(self, event):
        """Called when the thread finishes, successfully or not."""
        self._toggle_controls(is_running=False)
        self.worker = None

    def on_close(self, event):
        """Handles the window close event."""
        if self.worker and self.worker.is_alive():
            dlg = wx.MessageDialog(self,
                                   "A process is still running. Are you sure you want to exit?\n"
                                   "The process will be stopped.",
                                   "Confirm Exit", wx.OK | wx.CANCEL | wx.ICON_WARNING)
            if dlg.ShowModal() == wx.ID_OK:
                self.stop_event.set()
                self.worker.join()  # Wait for thread to finish
            else:
                dlg.Destroy()
                return  # Don't close
            dlg.Destroy()

        self._save_config()
        self.taskBarIcon.Destroy()
        self.Destroy()

    # --- Configuration and Widget Helpers ---
    def _load_config(self):
        """Loads configuration from a JSON file."""
        # These are the default values for a first-time startup
        defaults = {
            'master_dir': r"E:\Synchting\lablelsv2\Sensorbox Raw Lidar Data",
            'camera_dir': r"E:\Synchting\lablelsv2\Sensorbox Raw Camera Data\Test_2025-06-13",
            'output_dir': r"E:\backup Hairoad sync\juli 2025\synced_output5",
            'border': 100,
            'time_diff_threshold': 0.05
        }
        try:
            # If config.json exists, load it and update the defaults
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                defaults.update(config)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, just use the hardcoded defaults
            pass
        return defaults

    def _save_config(self):
        """Saves current configuration to a JSON file."""
        self._update_and_validate_config()
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def _update_and_validate_config(self):
        """Updates the config dict from UI and validates values."""
        self.config['master_dir'] = self.master_dir_txt.GetValue()
        self.config['camera_dir'] = self.camera_dir_txt.GetValue()
        self.config['output_dir'] = self.output_dir_txt.GetValue()
        try:
            self.config['border'] = int(self.border_txt.GetValue())
            self.config['time_diff_threshold'] = float(self.time_diff_txt.GetValue())
            return True
        except ValueError:
            wx.MessageBox("Invalid input for 'Border' or 'Time Diff'. Please enter numbers.", "Input Error",
                          wx.ICON_ERROR)
            return False

    def _toggle_controls(self, is_running):
        """Enable/disable controls based on processing state."""
        self.start_button.Enable(not is_running)
        self.stop_button.Enable(is_running)

    def _create_path_row(self, sizer, label_text, config_key):
        """Helper to create a label, text ctrl, and browse button row."""
        label = wx.StaticText(self.panel, label=label_text)
        text_ctrl = wx.TextCtrl(self.panel, value=str(self.config.get(config_key, "")))
        button = wx.Button(self.panel, label="Browse...")
        button.Bind(wx.EVT_BUTTON, lambda evt, tc=text_ctrl: self.on_browse(evt, tc))
        sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(button, 0)
        return text_ctrl

    def _create_text_row(self, sizer, label_text, config_key):
        """Helper to create a label and text ctrl row."""
        label = wx.StaticText(self.panel, label=label_text)
        text_ctrl = wx.TextCtrl(self.panel, value=str(self.config.get(config_key, "")))
        sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        sizer.Add((0, 0))  # Spacer
        return text_ctrl

    def on_browse(self, event, text_ctrl):
        """Opens a directory dialog."""
        with wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                text_ctrl.SetValue(dlg.GetPath())


if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()