#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wx
import os
import glob
import threading
import sys
from datetime import datetime


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller. """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# --- Real Processing Task ---
# The dummy function has been removed. We now import the actual function.
# Ensure that the 'hairoad_calib' library is installed and accessible in your
# Python environment where you run this script.
from hairoad_calib.timesync.timesync_crawler import process_master_file



# --- Custom Event Setup for Thread-Safe UI Updates ---
# Event for log text updates
myEVT_LOG_TYPE = wx.NewEventType()
EVT_LOG = wx.PyEventBinder(myEVT_LOG_TYPE, 1)

# NEW: Event for status bar updates
myEVT_STATUS_TYPE = wx.NewEventType()
EVT_STATUS = wx.PyEventBinder(myEVT_STATUS_TYPE, 1)


class LogEvent(wx.PyEvent):
    """Simple event to carry log data."""
    def __init__(self, data):
        wx.PyEvent.__init__(self)
        self.SetEventType(myEVT_LOG_TYPE)
        # Add a custom icon to the GUI

        self.data = data

# NEW: Event class for status updates
class StatusEvent(wx.PyEvent):
    """Simple event to carry status data."""
    def __init__(self, data):
        wx.PyEvent.__init__(self)
        self.SetEventType(myEVT_STATUS_TYPE)
        self.data = data


# Class to redirect stdout/stderr
class WxTextCtrlRedirector:
    def __init__(self, callback):
        self.output_callback = callback

    def write(self, string):
        # Using wx.CallAfter is safer for thread-to-GUI communication.
        wx.CallAfter(self.output_callback, string)

    def flush(self):
        pass


# --- Worker Thread ---
class ProcessingThread(threading.Thread):
    def __init__(self, parent, config, stop_event):
        threading.Thread.__init__(self)
        self._parent = parent
        self._config = config
        self._stop_event = stop_event
        self.was_stopped = False  # Flag to track if stop was user-initiated

    def run(self):
        def log_to_gui(message):
            evt = LogEvent(message)
            wx.PostEvent(self._parent, evt)

        # NEW: Helper to post status updates
        def update_status(message):
            wx.PostEvent(self._parent, StatusEvent(message))

        redirector = WxTextCtrlRedirector(log_to_gui)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = redirector
        sys.stderr = redirector

        try:
            master_dir = self._config['master_dir']
            camera_dir = self._config['camera_dir']
            output_dir = self._config['output_dir']
            border_size = self._config['border']
            # NEW: Get the new parameter from the config dict
            time_diff = self._config['time_diff_threshold']

            if not os.path.exists(output_dir):
                print(f"Output directory does not exist. Creating: {output_dir}")
                os.makedirs(output_dir)

            update_status("Searching for master files...")
            master_file_list = glob.glob(os.path.join(master_dir, "*.bin.txt"))
            total_files = len(master_file_list)

            if not master_file_list:
                print(f"❌ No master files found in '{master_dir}'. Please check the path.")
                update_status(f"No files found in '{master_dir}'.")
            else:
                print(f"Found {total_files} master files to process.")
                for i, master_file in enumerate(master_file_list):
                    if self._stop_event.is_set():
                        print("\n🛑 Processing stopped by user.")
                        self.was_stopped = True
                        update_status(f"Stopped by user.")
                        break

                    # NEW: Update status label in GUI with current file
                    status_msg = f"Processing file {i + 1} of {total_files}: {os.path.basename(master_file)}"
                    update_status(status_msg)

                    print(f"\n--- Processing file {i + 1} of {total_files} ---")
                    # NEW: Pass the new parameter to the processing function
                    process_master_file(
                        master_file,
                        camera_dir,
                        output_dir,
                        border=border_size,
                        TIME_DIFF_THRESHOLD=time_diff
                    )

                if not self.was_stopped:
                    print("\n\n✅ All master files have been processed.")
                    update_status("Processing complete.")

        except Exception as e:
            import traceback
            print(f"\n\n❌ An unhandled error occurred: {e}")
            print(traceback.format_exc())
            update_status(f"Error occurred: {e}")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_to_gui("##THREAD_FINISHED##")


# --- Main GUI Frame ---
class MainFrame(wx.Frame):
    def __init__(self):
        # Increased size slightly for new fields
        super().__init__(None, title="Hairoad Data Processor", size=(750, 650))

        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.config = {
            'master_dir': r"E:\Synchting\lablelsv2\Sensorbox Raw Lidar Data",
            'camera_dir': r"E:\Synchting\lablelsv2\Sensorbox Raw Camera Data\Test_2025-06-13",
            'output_dir': r"E:\backup Hairoad sync\juli 2025\synced_output5",
            'border': 100,
            # NEW: Added default config parameter
            'time_diff_threshold': 0.05
        }

        # --- Input Fields ---
        grid_sizer = wx.FlexGridSizer(cols=3, gap=(5, 5))
        grid_sizer.AddGrowableCol(1)
        self.master_dir_txt = self.create_path_row(grid_sizer, "Master Files Directory:", 'master_dir')
        self.camera_dir_txt = self.create_path_row(grid_sizer, "Camera Data Directory:", 'camera_dir')
        self.output_dir_txt = self.create_path_row(grid_sizer, "Base Output Directory:", 'output_dir')

        # Border size input
        border_label = wx.StaticText(self.panel, label="Border Size:")
        self.border_txt = wx.TextCtrl(self.panel, value=str(self.config['border']))
        grid_sizer.Add(border_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        grid_sizer.Add(self.border_txt, 1, wx.EXPAND)
        grid_sizer.Add((0, 0))  # Spacer

        # NEW: TIME_DIFF_THRESHOLD input field
        time_diff_label = wx.StaticText(self.panel, label="Time Diff Threshold (s):")
        self.time_diff_txt = wx.TextCtrl(self.panel, value=str(self.config['time_diff_threshold']))
        grid_sizer.Add(time_diff_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        grid_sizer.Add(self.time_diff_txt, 1, wx.EXPAND)
        grid_sizer.Add((0, 0))  # Spacer

        main_sizer.Add(grid_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # --- Buttons ---
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.start_button = wx.Button(self.panel, label="Start Processing")
        self.stop_button = wx.Button(self.panel, label="Stop")
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)
        self.stop_button.Disable()
        button_sizer.Add(self.start_button, 0, wx.ALL, 5)
        button_sizer.Add(self.stop_button, 0, wx.ALL, 5)
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER)


        try:
            icon_path = resource_path("Invilab_ico.ico")
            if os.path.exists(icon_path):
                self.SetIcon(wx.Icon(icon_path, wx.BITMAP_TYPE_ICO))
        except Exception as e:
            print(f"Icon not found or failed to load: {e}")

        # NEW: Status Display Label
        self.status_label = wx.StaticText(self.panel, label="Status: Idle")
        font = self.status_label.GetFont()
        font.SetStyle(wx.FONTSTYLE_ITALIC)
        self.status_label.SetFont(font)
        main_sizer.Add(self.status_label, 0, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 5)

        # --- Log Window ---
        log_label = wx.StaticText(self.panel, label="Log Output:")
        main_sizer.Add(log_label, 0, wx.LEFT | wx.TOP, 10)
        self.log_txt = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        main_sizer.Add(self.log_txt, 1, wx.EXPAND | wx.ALL, 10)

        self.panel.SetSizer(main_sizer)
        self.Centre()
        self.Show()

        self.Bind(EVT_LOG, self.on_log_result)
        self.Bind(EVT_STATUS, self.on_status_update) # NEW: Bind status event
        self.worker = None
        self.stop_event = None

    def create_path_row(self, sizer, label_text, config_key):
        label = wx.StaticText(self.panel, label=label_text)
        text_ctrl = wx.TextCtrl(self.panel, value=self.config[config_key])
        browse_button = wx.Button(self.panel, label="Browse...")
        browse_button.Bind(wx.EVT_BUTTON, lambda evt, key=config_key, tc=text_ctrl: self.on_browse(evt, key, tc))
        sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer.Add(text_ctrl, 1, wx.EXPAND)
        sizer.Add(browse_button, 0)
        return text_ctrl

    def on_browse(self, event, config_key, text_ctrl):
        dlg = wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            text_ctrl.SetValue(path)
            self.config[config_key] = path
        dlg.Destroy()

    def on_start(self, event):
        self.log_txt.Clear()
        self.log_txt.AppendText(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting process...\n")
        self.status_label.SetLabel("Status: Initializing...")

        # Update config from text fields
        self.config['master_dir'] = self.master_dir_txt.GetValue()
        self.config['camera_dir'] = self.camera_dir_txt.GetValue()
        self.config['output_dir'] = self.output_dir_txt.GetValue()

        try:
            self.config['border'] = int(self.border_txt.GetValue())
        except ValueError:
            wx.MessageBox("Invalid border size. Please enter an integer.", "Error", wx.ICON_ERROR)
            self.status_label.SetLabel("Status: Idle")
            return

        # NEW: Get and validate the time diff threshold
        try:
            self.config['time_diff_threshold'] = float(self.time_diff_txt.GetValue())
        except ValueError:
            wx.MessageBox("Invalid Time Diff Threshold. Please enter a number (e.g., 0.05).", "Error", wx.ICON_ERROR)
            self.status_label.SetLabel("Status: Idle")
            return

        self.start_button.Disable()
        self.stop_button.Enable()

        self.stop_event = threading.Event()
        self.worker = ProcessingThread(self, self.config, self.stop_event)
        self.worker.start()

    def on_stop(self, event):
        if self.worker and self.stop_event:
            self.status_label.SetLabel("Status: Sending stop signal...")
            self.log_txt.AppendText("\n--- Sending stop signal... ---\n")
            self.stop_event.set()
            self.stop_button.Disable()

    # NEW: Event handler for status updates
    def on_status_update(self, event):
        """Receives status messages from the worker thread."""
        self.status_label.SetLabel(f"Status: {event.data}")

    def on_log_result(self, event):
        """Receives log messages from the worker thread and updates the GUI."""
        if event.data == "##THREAD_FINISHED##":
            # Final status is set by the thread itself or on_stop, so we just reset buttons
            self.start_button.Enable()
            self.stop_button.Disable()
            self.worker = None
            return

        text = event.data

        # This logic handles the overwriting progress line (starts with \r)
        if text.startswith('\r'):
            text_to_write = text.lstrip('\r')
            # Get the starting character position of the very last line in the control
            last_line_start_pos = self.log_txt.XYToPosition(0, self.log_txt.GetNumberOfLines() - 1)
            # Replace the content of the last line with the new progress text
            self.log_txt.Replace(last_line_start_pos, self.log_txt.GetLastPosition(), text_to_write)
        else:
            # This is a normal line of text, just append it.
            self.log_txt.AppendText(text)


if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()