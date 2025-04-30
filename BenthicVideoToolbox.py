import sys
import re
import components.scripts as scripts
import platform
import pathlib as pl
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.ttk as ttk
try:
    from tkhtmlview import HTMLLabel
except ImportError:
    pass
try:
    import wslPath
except ImportError:
    pass
try:
    import tkinterDnD
except ImportError:
    pass
from assets.icon import icon_data

if 'tkinterDnD' in sys.modules:
    class Application(tkinterDnD.Tk):
        def __init__(self):
            tkinterDnD.Tk.__init__(self)
            self.tabControl = ttk.Notebook(self)
            tab0 = PreprocessPage(self.tabControl, self)
            tab1 = PostprocessPage(self.tabControl, self)
            tab2 = BiigleAPIPage(self.tabControl, self)
            self.tabControl.add(tab0, text="Preprocess")
            self.tabControl.add(tab1, text="Postprocess")
            self.tabControl.add(tab2, text="Biigle API login")
            # self.tabControl.tab(tab1, state = "disabled")
            self.tabControl.pack(side="top", expand=True, fill="both")
else:
    class Application(tk.Tk):
        def __init__(self):
            tk.Tk.__init__(self)
            self.tabControl = ttk.Notebook(self)
            tab0 = PreprocessPage(self.tabControl, self)
            tab1 = PostprocessPage(self.tabControl, self)
            tab2 = BiigleAPIPage(self.tabControl, self)
            self.tabControl.add(tab0, text="Preprocess")
            self.tabControl.add(tab1, text="Postprocess")
            self.tabControl.add(tab2, text="Biigle API login")
            # self.tabControl.tab(tab1, state = "disabled")
            self.tabControl.pack(side="top", expand=True, fill="both")

def convert_path(path):
    if path:
        if path[0] == '"':
            path = path[1:]
        if path[-1] == '"':
            path = path[:-1]
    if(platform.system() == "Linux" and wslPath.is_windows_path(path)):
        path = wslPath.to_posix(path)
    elif(platform.system() == "Windows"):
        if bool(re.match(r"^[A-Za-z]:/", path)):
            path = path.replace("/", "\\")
        elif wslPath.is_posix_path(path):
            path = wslPath.to_windows(path)
    return path

def get_video_path(page, event=None):
    page.video_path = convert_path(page.video_entry.get())
    page.video_entry.delete(0, "end")
    page.video_entry.insert('end', page.video_path)

def get_cut_video_path(page, event=None):
    page.cut_video_path = convert_path(page.cut_video_entry.get())
    page.cut_video_entry.delete(0, "end")
    page.cut_video_entry.insert('end', page.cut_video_path)

def update_path(path, entry, event=None):
    path = convert_path(path)
    entry.delete(0, "end")
    entry.insert('end', path)
    return path

def get_nav_path(page, event=None):
    page.nav_path = convert_path(page.nav_entry.get())
    page.nav_entry.delete(0, "end")
    page.nav_entry.insert('end', page.nav_path)

def get_csv_path(page, event=None):
    page.csv_path = convert_path(page.csv_entry.get())
    page.csv_entry.delete(0, "end")
    page.csv_entry.insert('end', page.csv_path)

def get_user_metadata_path(page, event=None):
    print("get_user_metadata_path {}".format(page.user_metadata_entry.get()))
    page.user_metadata_path = convert_path(page.user_metadata_entry.get())
    page.user_metadata_entry.delete(0, "end")
    page.user_metadata_entry.insert('end', page.user_metadata_path)

def drop(stringvar, event):
    if(platform.system() == "Windows"):
        s = event.data.replace("/", "\\")
    else:
        s = event.data
    stringvar.set(s)

class BiigleAPIPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, width=200, height= 400)
        self.pack(expand=True, fill="both", padx=20, pady=20)
        # page label
        self.page_label = ttk.Label(self, text="Login to Biigle Rest API")
        self.page_label.pack(pady=10)
        # some text to inform user
        try:
            self.info_label = HTMLLabel(self, height=2, html='<p style="font-size:10px">Login to Biigle Rest API using your email and API token (you can generate one inside Biigle. Go to settings > Tokens.) See '
                                                '<a href="https://calcul01.epoc.u-bordeaux.fr:8443/doc/api/index.html"> documentation </a>'
                                                'for more details.</p>')
        except:
            self.info_label = ttk.Label(self, text="Login to Biigle Rest API using your email and API token (you can generate one inside Biigle. Go to settings > Tokens.) See https://calcul01.epoc.u-bordeaux.fr:8443/doc/api/index.html documentation for more details")
        self.info_label.pack(side="top", padx=10, pady=10)
        # login frame
        self.login_frame = ttk.Frame(self)
        self.login_frame.pack(padx=5, pady=20)
        # email section
        self.email_frame = ttk.Frame(self.login_frame)
        self.email_frame.pack(side="left", fill="x", padx=5)
        self.email_label = ttk.Label(self.email_frame, text="Biigle user email:")
        self.email_label.pack(side="left", padx=2)
        self.email_entry = ttk.Entry(self.email_frame, width=25)
        self.email_entry.pack(fill="x", padx=2)
        # token section
        self.token_frame = ttk.Label(self.login_frame)
        self.token_frame.pack(padx=5)
        self.token_label = ttk.Label(self.token_frame, text="Biigle API token:")
        self.token_label.pack(side="left", fill="x", padx=2)
        self.token_entry = ttk.Entry(self.token_frame, width=25)
        self.token_entry.pack(fill="x", padx=2)

class PreprocessPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.pack(expand=True, fill="both", padx=20, pady=20)
        self.page_label = ttk.Label(self, text="Preprocessing data")
        self.page_label.pack(pady=10)

        self.video_string = tk.StringVar()
        self.video_path = None
        self.video_entry_frame = ttk.Frame(self)
        self.video_entry_frame.pack(padx=10, pady=10, fill="x")
        self.video_entry_label = ttk.Label(self.video_entry_frame, text="Source video file:")
        self.video_entry_label.pack(side="left", padx=2)
        if 'tkinterDnD' in sys.modules:
            self.video_entry = ttk.Entry(self.video_entry_frame, ondrop=lambda event: drop(self.video_string, event), text=self.video_string)
        else:
            self.video_entry = ttk.Entry(self.video_entry_frame, text=self.video_string)
        self.video_entry.pack(fill="x", padx=2)
        if 'wslPath' in sys.modules:
            self.video_entry.bind("<FocusOut>", lambda event: update_path(self.video_entry.get(), self.video_entry))

        self.cut_video_string = tk.StringVar()
        self.cut_video_path = None
        self.cut_video_entry_frame = ttk.Frame(self)
        self.cut_video_entry_frame.pack(padx=10, pady=10, fill="x")
        self.cut_video_entry_label = ttk.Label(self.cut_video_entry_frame, text="Cut video file:")
        self.cut_video_entry_label.pack(side="left", padx=2)
        if 'tkinterDnD' in sys.modules:
            self.cut_video_entry = ttk.Entry(self.cut_video_entry_frame, ondrop=lambda event: drop(self.cut_video_string, event), text=self.cut_video_string)
        else:
            self.cut_video_entry = ttk.Entry(self.cut_video_entry_frame, text=self.cut_video_string)
        self.cut_video_entry.pack(fill="x", padx=2)
        if 'wslPath' in sys.modules:
            self.cut_video_entry.bind("<FocusOut>", lambda event: update_path(self.cut_video_entry.get(), self.cut_video_entry))

        self.nav_string = tk.StringVar()
        self.nav_entry_frame = ttk.Frame(self)
        self.nav_entry_frame.pack(fill="x", padx=10, pady=10) #expand=True,
        self.nav_entry_label = ttk.Label(self.nav_entry_frame, text="Navigation file:")
        self.nav_entry_label.pack(side="left", padx=2)
        if 'tkinterDnD' in sys.modules:
            self.nav_entry = ttk.Entry(self.nav_entry_frame, ondrop=lambda event: drop(self.nav_string, event), text=self.nav_string)
        else:
            self.nav_entry = ttk.Entry(self.nav_entry_frame, text=self.nav_string)
        self.nav_entry.pack(fill="x", padx=2)
        if 'wslPath' in sys.modules:
            self.nav_entry.bind("<FocusOut>", lambda event: update_path(self.nav_entry.get(), self.nav_entry))

        # cut video section
        self.cut_frame = ttk.Frame(self)
        self.cut_frame.pack(anchor="e", padx=10, pady=10)

        self.example_text = "HH:MM:SS"
        self.cut_from_frame = ttk.Frame(self.cut_frame)
        self.cut_from_frame.pack(side="left", padx=2)
        self.cut_from_label = ttk.Label(self.cut_from_frame, text="From:")
        self.cut_from_label.pack(side="left", padx=2)
        self.cut_from_entry = ttk.Entry(self.cut_from_frame, foreground="#A9A9A9")
        self.cut_from_entry.insert(0, self.example_text)
        self.cut_from_entry.pack(side="left", fill="x", padx=2)
        self.cut_from_entry.bind("<FocusIn>", lambda event, e=self.cut_from_entry: self.remove_example_cb(e))
        self.cut_from_entry.bind("<FocusOut>", lambda event, e=self.cut_from_entry, t=self.example_text: self.reset_example_cb(e, t))

        self.cut_to_frame = ttk.Frame(self.cut_frame)
        self.cut_to_frame.pack(side="left", padx=2)
        self.cut_to_label = ttk.Label(self.cut_to_frame, text="To:")
        self.cut_to_label.pack(side="left", padx=2)
        self.cut_to_entry = ttk.Entry(self.cut_to_frame, foreground="#A9A9A9")
        self.cut_to_entry.insert(0, self.example_text)
        self.cut_to_entry.pack(side="left", fill="x", padx=2)
        self.cut_to_entry.bind("<FocusIn>", lambda event, e=self.cut_to_entry: self.remove_example_cb(e))
        self.cut_to_entry.bind("<FocusOut>", lambda event, e=self.cut_to_entry, t=self.example_text: self.reset_example_cb(e, t))

        self.cut_video_button = ttk.Button(self.cut_frame, text="Cut video", command=self.cut_video)
        self.cut_video_button.pack(padx=20)

        # convert nav file section
        self.convert_frame = ttk.Frame(self)
        self.convert_frame.pack(anchor="e", padx=10, pady=5)
        self.convert_nav_label = ttk.Label(self.convert_frame, text="Convert Pagure's nav file to CSV metadata file for use inside Biigle:")
        self.convert_nav_label.pack(side="left", padx=2)
        self.convert_nav_button = ttk.Button(self.convert_frame, text="Convert", command=self.convert_nav_to_csv)
        self.convert_nav_button.pack(padx=20)

        # load into Biigle
        self.load_frame = ttk.Frame(self)
        self.load_frame.pack(anchor="e", padx=10, pady=10)
        self.load_label = ttk.Label(self.load_frame, text="Load metadata file into Biigle with volume id:")
        self.load_label.pack(side="left", padx=2)
        self.volume_id_entry = ttk.Entry(self.load_frame, width=10)
        self.volume_id_entry.pack(padx=20)

        # user metadata file path
        self.user_metadata_string = tk.StringVar()
        self.user_metadata_path = None
        self.user_metadata_frame = ttk.Frame(self)
        self.user_metadata_frame.pack(side="bottom", padx=10, pady=20, fill="x")
        self.user_metadata_label = ttk.Label(self.user_metadata_frame, text="BVT metadata file:")
        self.user_metadata_label.pack(side="left", padx=2)
        if 'tkinterDnD' in sys.modules:
            self.user_metadata_entry = ttk.Entry(self.user_metadata_frame, ondrop=lambda event: drop(self.user_metadata_string, event), text=self.user_metadata_string)
        else:
            self.user_metadata_entry = ttk.Entry(self.user_metadata_frame, text=self.user_metadata_string)
        self.user_metadata_entry.pack(fill="x", padx=2)
        if 'wslPath' in sys.modules:
            self.user_metadata_entry.bind("<FocusOut>", lambda event: update_path(self.user_metadata_entry.get(), self.user_metadata_entry))

    def remove_example_cb(self, entry, event=None):
        if str(entry.cget("foreground")) == "#A9A9A9":
            entry.delete(0, "end")
            entry.configure(foreground="black")

    def reset_example_cb(self, entry, text, event=None):
        if not entry.get():
            entry.insert(0, text)
            entry.configure(foreground="#A9A9A9")

    def entry_cb(self, message):
        window = entryWindow(self, "Enter value", message)
        self.wait_window(window.top)
        return window.value

    def cut_video(self):
        self.video_path = self.video_entry.get()
        self.nav_path = self.nav_entry.get()
        self.user_metadata_path = self.user_metadata_string.get()
        if (not self.video_path):
            self.video_path = filedialog.askopenfilename(parent=self, title="Choose a video file to cut", filetypes=[('all', '*'), ('avi videos', '*.avi'), ('mp4 videos', '*.mp4')])
        p = pl.Path(self.video_path)
        if not p.exists() or not p.is_file():
            messagebox.showerror("Error", "{} is not a regular file. Please provide a valid file path.".format(self.video_path))
        if not self.user_metadata_path:
            self.user_metadata_path = filedialog.asksaveasfilename(parent=self, title="Select or create BVT metadata file", filetypes=[('csv files', '*.csv'), ('text files', '*.txt')], defaultextension='.csv')
            self.user_metadata_entry.delete(0, "end")
            self.user_metadata_entry.insert(0, self.user_metadata_path)
        if (not self.nav_path or len(self.nav_path) == 0):
            if (str(self.cut_to_entry.cget("foreground")) or str(self.cut_from_entry.cget("foreground"))) == "#A9A9A9" or not self.cut_from_entry.get() or not self.cut_to_entry.get():
                messagebox.showerror(title="Error: ", message="If no navigation file is provided, you need to specify start and end cut times (in seconds or with format HH:MM:SS) to cut video file.")
                return
            else:
                start = str(self.cut_from_entry.get())
                end = str(self.cut_to_entry.get())
                if not scripts.test_time_format(start) or not scripts.test_time_format(end):
                    messagebox.showerror(title="Error: ", message="Wrong format for start_value/end values. Please use seconds or HH:MM:SS.")
                    return
        else:
            try:
                t0, start, end, start_abs, end_abs = scripts.read_cut_times_from_nav(self.nav_path, self.user_metadata_path)
            except:
                return
            lines = ["The program found the following cut times (relative) in navigation file:", "{} and {} corresponding to absolute timestamps {} and {}".format(start, end, start_abs, end_abs), "Do you want to proceed ?"]
            if not (messagebox.askyesno(title="Confirm cut times", message="\n".join(lines))):
                window = entryWindow(self, "Time cut start", "Enter the 'start cut time' (relative) in seconds or with format HH:MM:SS:")
                self.wait_window(window.top)
                start = window.value
                window = entryWindow(self, "Time cut end", "Enter the 'end cut time' (relative) in seconds or with format HH:MM:SS:")
                self.wait_window(window.top)
                end = window.value
                if not start or not end:
                    return
            self.cut_from_entry.delete(0, "end")
            self.cut_from_entry.insert(0, start)
            self.cut_to_entry.delete(0, "end")
            self.cut_to_entry.insert(0, end)
        output_path = filedialog.asksaveasfilename(parent=self, title="Save as", filetypes=[('mp4 videos', '*.mp4'), ('avi videos', '*.avi'), ('mpeg videos', '*.mpeg'),
                                                    ('quicktime videos', '*.mov'), ('all files', '*')], defaultextension='.mp4')
        output_path = update_path(output_path, self.cut_video_entry)
        result = scripts.cut_command(self.video_path, start, end, output_path, self.user_metadata_path)
        if result == 0:
            messagebox.showinfo("Success", "Video {} has been successfully cut and has been saved to {}".format(p.name, output_path))
        else:
            messagebox.showerror("Error", "Operation failed, please retry.")

    def convert_nav_to_csv(self):
        video_path = self.cut_video_string.get()
        self.nav_path = self.nav_entry.get()
        self.user_metadata_path = self.user_metadata_string.get()
        if not self.nav_path:
            self.nav_path = filedialog.askopenfilename(title="Choose a Pagure navigation file to convert", filetypes=[('text files', '*.txt')])
            if not self.nav_path:   # if user cancelled command
                return
        if not self.user_metadata_path:
            self.user_metadata_path = filedialog.asksaveasfilename(parent=self, title="Select or create BVT metadata file", filetypes=[('csv files', '*.csv'), ('text files', '*.txt')], defaultextension='.csv')
            self.user_metadata_entry.delete(0, "end")
            self.user_metadata_entry.insert(0, self.user_metadata_path)
        apiTab = app.tabControl.nametowidget(app.tabControl.tabs()[2])
        email = apiTab.email_entry.get()
        token = apiTab.token_entry.get()
        volume_id = self.volume_id_entry.get()
        if self.volume_id_entry.get() and (not email or not token):
            messagebox.showerror(title="Error: ", message="To connect to Biigle API, please fill in the login details inside 'Biigle API Login' tab.")
            return
        if not video_path:
            source = self.video_entry.get()
            videoname = pl.Path(source).name
            if messagebox.askyesno(message="Was the video {} cut before being annotated ?".format(videoname)):
                window = entryWindow(self, "Video name", "Enter the (cut) video full path associated to this navigation file:")
                self.wait_window(window.top)
                video_path = update_path(window.value, self.cut_video_entry)
            else:
                video_path = source
            if not video_path:
                return
        s = pl.Path(video_path).suffix
        if s and s != ".mp4":
            if not messagebox.askyesno("Warning", "Video extension is {}, is it the same as the file loaded on Biigle ? (Note that cut movies are converted to .mp4)".format(s)):
                w = entryWindow(self, "Video extension", "Enter the video extension of the uploaded file on Biigle:")
                self.wait_window(w.top)
                s = w.value
                if not s:
                    return
                try:
                    video_path = pl.Path(video_path).with_suffix(s)
                except ValueError as e:
                    messagebox.showerror("Error", e)
                    return
        output_path = filedialog.asksaveasfilename(parent=self, title="Save as", initialdir=pl.Path(self.nav_path).parent, filetypes=[('csv files', '*.csv'), ('all files', '*')], defaultextension='.csv')
        if not output_path:
            return
        result = scripts.convert_nav_to_csv(self.nav_path, video_path, self.entry_cb, self.user_metadata_path, output_path, True, volume_id, email, token)
        if result:
            messagebox.showinfo("Success", "Metadata file has been written to {}".format(output_path))
        else:
            messagebox.showerror(title="Error", message="Conversion failed, please retry.")

class PostprocessPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.pack(expand=True, fill="both", padx=20, pady=20)
        self.page_label = ttk.Label(self, text="Postprocessing data")
        self.page_label.pack(pady=10)

        # biigle to yolo images annotations section
        self.csv_string = tk.StringVar()
        self.csv_entry_frame = ttk.Frame(self)
        self.csv_entry_frame.pack(fill="x", padx=10, pady=10)
        self.csv_entry_label = ttk.Label(self.csv_entry_frame, text="Biigle video annotation file:")
        self.csv_entry_label.pack(side="left", padx=2)
        if 'tkinterDnD' in sys.modules:
            self.csv_entry = ttk.Entry(self.csv_entry_frame, ondrop=lambda event: drop(self.csv_string, event), text=self.csv_string)
        else:
            self.csv_entry = ttk.Entry(self.csv_entry_frame, text=self.csv_string)
        self.csv_entry.pack(fill="x", padx=2)
        self.csv_entry.bind("<FocusOut>", lambda event: get_csv_path(self))

        self.biigle_to_yolo_frame = ttk.Frame(self)
        self.biigle_to_yolo_frame.pack(anchor="e", padx=10, pady=10)
        self.biigle_to_yolo_label = ttk.Label(self.biigle_to_yolo_frame, text="Convert Biigle video annotation file to YOLO-formatted images annotations files:")
        self.biigle_to_yolo_label.pack(side="left")
        self.biigle_to_yolo_button = ttk.Button(self.biigle_to_yolo_frame, text="Convert", command=self.biigle_to_yolo)
        self.biigle_to_yolo_button.pack(padx=20)

        # detect laserpoints section
        self.laser_tracks = None
        self.laserpoints_frame = ttk.Frame(self, relief=tk.GROOVE)
        self.laserpoints_frame.pack(anchor="e", padx=10, pady=10, fill="both")
        self.laserpoints_label = ttk.Label(self.laserpoints_frame, text="Detect laserpoints inside video")
        self.laserpoints_label.pack(pady=10)

        self.mode_frame = ttk.Frame(self.laserpoints_frame)
        self.mode_frame.pack(anchor="w", padx=10, pady=10)
        self.mode_label = ttk.Label(self.mode_frame, text="Detection mode:")
        self.mode_label.pack(side="left", padx=10)
        self.mode_combobox = ttk.Combobox(self.mode_frame, values=["manual", ""], state='readonly', width=10)
        self.mode_combobox.pack(side="left")
        self.mode_combobox.bind("<<ComboboxSelected>>", self.laser_mode_widgets)

        self.manual_mode_frame = ttk.Frame(self.laserpoints_frame)
        self.laser_label = ttk.Label(self.manual_mode_frame, text="Label used to annotate laserpoints:")
        self.laser_label.pack(side="left", padx=10)
        self.laser_label_entry = ttk.Entry(self.manual_mode_frame, width=10)
        self.laser_label_entry.pack(side="left")

        self.detect_button = ttk.Button(self.laserpoints_frame, text="Detect", command=self.detect_laserpoints, state="disabled")
        self.detect_button.pack(anchor="se", side="right", padx=20, pady=20)

        # build eco profiler export section
        self.eco_profiler_frame = ttk.Frame(self, relief=tk.GROOVE)
        self.eco_profiler_frame.pack(anchor="e", padx=10, pady=10, fill="both")
        self.eco_profiler_label = ttk.Label(self.eco_profiler_frame, text="Build ecological profiler export")
        self.eco_profiler_label.pack(pady=10)

        self.entries_frame = ttk.Frame(self.eco_profiler_frame)
        self.entries_frame.pack(anchor="w", padx=10, pady=10, fill="x")
        self.eco_laser_dist_label = ttk.Label(self.entries_frame, text="Distance between lasers in cm:")
        self.eco_laser_dist_label.pack(side="left", padx=10)
        self.eco_laser_dist_entry = ttk.Entry(self.entries_frame, width=10)
        self.eco_laser_dist_entry.pack(side="left", padx=[0, 10])

        self.eco_threshold_label = ttk.Label(self.entries_frame, text="dy_max to measure annotations:\n(in % of video's height)")
        self.eco_threshold_label.pack(side="left", padx=15)
        self.eco_threshold_entry = ttk.Entry(self.entries_frame, width=10)
        self.eco_threshold_entry.insert(0, '10')
        self.eco_threshold_entry.pack(side="left")

        self.annot_mode_frame = ttk.Frame(self.eco_profiler_frame)
        self.annot_mode_frame.pack(anchor="w", padx=10, fill="x")
        self.annot_mode_label = ttk.Label(self.annot_mode_frame, text="Mode used to annotate video:")
        self.annot_mode_label.pack(side="left", padx=10)
        self.annot_mode_combobox = ttk.Combobox(self.annot_mode_frame, values=["full", "sampled"], state='readonly', width=10)
        self.annot_mode_combobox.current(0)
        self.annot_mode_combobox.pack(side="left")
        self.annot_mode_combobox.bind("<<ComboboxSelected>>", self.annot_mode_widgets)

        self.sampled_mode_frame = ttk.Frame(self.annot_mode_frame)
        self.sampled_markers_label = ttk.Label(self.sampled_mode_frame ,text="Labels used to delimit annotation sections:")
        self.sampled_markers_label.pack(anchor="w", padx=10, pady=5)
        self.start_marker_label = ttk.Label(self.sampled_mode_frame, text="Start:")
        self.start_marker_label.pack(side="left", padx=10)
        self.start_marker_entry = ttk.Entry(self.sampled_mode_frame, width=12)
        self.start_marker_entry.pack(side="left")
        self.stop_marker_label = ttk.Label(self.sampled_mode_frame, text="Stop:")
        self.stop_marker_label.pack(side="left", padx=10)
        self.stop_marker_entry = ttk.Entry(self.sampled_mode_frame, width=12)
        self.stop_marker_entry.pack(side="left")

        self.eco_profiler_button = ttk.Button(self.eco_profiler_frame, text="Export", command=self.eco_profiler)
        self.eco_profiler_button.pack(side="right", padx=20, pady=20)

    def biigle_to_yolo(self):
        csv_path = self.csv_entry.get()
        preprocessTab = app.tabControl.nametowidget(app.tabControl.tabs()[0])
        video_path = preprocessTab.cut_video_string.get()
        user_metadata_path = preprocessTab.user_metadata_string.get()
        if not video_path:
            source = preprocessTab.video_entry.get()
            if source:
                videoname = pl.Path(source).name
                if messagebox.askyesno(message="Was the video {} cut before being annotated ?".format(videoname)):
                    window = entryWindow(self, "Video name", "Enter the (cut) video full path associated to this annotation file:")
                    self.wait_window(window.top)
                    video_path = update_path(window.value, preprocessTab.cut_video_entry)
                else:
                    video_path = source
        if pl.Path(video_path).suffix != ".mp4":
            if not messagebox.askyesno("Warning", "Does video filepath {} corresponds to video on which the annotations had been processed (cut video if so) referrenced in annotation file ?".format(video_path)):
                video_paths = filedialog.askopenfilenames(title="Select the input video file(s) on which the annotations had been processed", filetypes=[('mp4 files', '*.mp4'), ('avi files', '*.avi')])
        if not video_path:
            video_paths = filedialog.askopenfilenames(title="Select the input video file(s) on which the annotations had been processed", filetypes=[('mp4 files', '*.mp4'), ('avi files', '*.avi')])
            if not video_paths:
                return
        else:
            video_paths = [video_path]
        if not user_metadata_path:
            user_metadata_path = filedialog.asksaveasfilename(parent=self, title="Select or create BVT metadata file", filetypes=[('csv files', '*.csv'), ('text files', '*.txt')], defaultextension='.csv')
            preprocessTab.user_metadata_entry.delete(0, "end")
            preprocessTab.user_metadata_entry.insert(0, user_metadata_path)
        output_path = filedialog.askdirectory(parent=self, title="Save as", mustexist=True)
        if not output_path:
            messagebox.showerror(title="Error", message="Conversion failed, please retry.")
            return
        scripts.biigle_annot_to_yolo(csv_path, user_metadata_path, video_paths, output_path)

    def laser_mode_widgets(self, event=None):
        mode = self.mode_combobox.get()
        if mode == 'manual':
            self.manual_mode_frame.pack(anchor="w", padx=10)
            self.detect_button["state"] = "normal"
        elif mode == "":
            self.manual_mode_frame.pack_forget()

    def annot_mode_widgets(self, event=None):
        mode = self.annot_mode_combobox.get()
        if mode == "full":
            self.sampled_mode_frame.pack_forget()
        elif mode == "sampled":
            self.sampled_mode_frame.pack(side="left", padx=10)

    def detect_laserpoints(self):
        mode = self.mode_combobox.get()
        if mode == "manual":
            label = self.laser_label_entry.get()
            if not label:
                messagebox.showerror("Error", "Please enter the lasers annotation label.")
                return
            csv_path = self.csv_entry.get()
            if not csv_path:
                csv_path = filedialog.askopenfilename(title="Select biigle annotation file with laserpoints annotations", filetypes=[('csv files', '*.csv')])
                if not csv_path:       # user cancelled command
                    return
            self.laser_tracks = scripts.manual_detect_laserpoints(label, csv_path)
        else:
            messagebox.showerror("Error", "Invalid laserpoints detection mode.")

        if self.laser_tracks:
            messagebox.showinfo("Success", "Laserpoints have been successfully detected, you can proceed to ecological export.")
        else:
            messagebox.showerror("Error", "Operation failed, please retry.")

    def eco_profiler(self):
        csv_path = self.csv_entry.get()
        if not csv_path:
            csv_path = filedialog.askopenfilename(title="Select a CSV video annotation file for this dataset", filetypes=[("csv files", "*.csv")])
        preprocessTab = app.tabControl.nametowidget(app.tabControl.tabs()[0])
        nav_path = preprocessTab.nav_entry.get()
        video_path = preprocessTab.cut_video_string.get()
        user_metadata_path = preprocessTab.user_metadata_string.get()
        # if cut video file not filled try to get source video file
        if not video_path:
            source = preprocessTab.video_entry.get()
            if source:
                videoname = pl.Path(source).name
                if messagebox.askyesno(message="Was the video {} cut before being annotated ?".format(videoname)):
                    window = entryWindow(self, "Video name", "Enter the (cut) video full path associated to this annotation file:")
                    self.wait_window(window.top)
                    video_path = update_path(window.value, preprocessTab.cut_video_entry)
                else:
                    video_path = source

        output_path = filedialog.asksaveasfilename(parent=self, title="Save as", initialdir=pl.Path(csv_path).parent, filetypes=[("csv files", "*.csv"), ('text files', '*.txt')])
        if not output_path: # or not nav_paths:
            messagebox.showerror(title="Error", message="Operation failed, please retry.")
            return
        if not user_metadata_path:
            user_metadata_path = filedialog.asksaveasfilename(parent=self, title="Select or create BVT metadata file", filetypes=[('csv files', '*.csv'), ('text files', '*.txt')], defaultextension='.csv')
            preprocessTab.user_metadata_entry.delete(0, "end")
            preprocessTab.user_metadata_entry.insert(0, user_metadata_path)

        threshold = self.eco_threshold_entry.get()
        if not threshold:
            window = entryWindow(self, "dy_max value", "Please enter the threshold distance between lasers and annotation:")
            self.wait_window(window.top)
            threshold = window.value
            if not threshold:   # if user cancelled command
                return

        start_sample, stop_sample = None, None
        annot_mode = self.annot_mode_combobox.get()
        if annot_mode == "sampled":
            start_sample = self.start_marker_entry.get()
            stop_sample = self.stop_marker_entry.get()
            if not start_sample or not stop_sample:
                messagebox.showerror("Error", "Please provide start and stop markers used in annotation sampling protocol.")
                return
        if not self.laser_tracks:
            if messagebox.askyesno(title="Info", message="Laserpoints have not been detected yet, do you want to export eco profiler anyway ? (Without size measurement)"):
                result = scripts.eco_profiler(csv_path, threshold, preprocessTab.entry_cb, user_metadata_path, video_path, nav_path, start_label=start_sample, stop_label=stop_sample, outPath=output_path)
            else:
                messagebox.showerror("Error", "Export cancelled, please proceed to laser detection first.")
                return
        else:
            laser_label = self.laser_label_entry.get()
            if not laser_label:
                window = entryWindow(self, "Laser annotation label", "Enter the label used to annotate lasers in biigle:")
                self.wait_window(window.top)
                laser_label = window.value
                if not laser_label:   # if user cancelled command
                    return
            laser_dist = self.eco_laser_dist_entry.get()
            if not laser_dist:
                window = entryWindow(self, "Distance between lasers", "Enter the distance between lasers in cm (see info file):")
                self.wait_window(window.top)
                laser_dist = window.value
                if not laser_dist:   # if user cancelled command
                    return
            result = scripts.eco_profiler(csv_path, threshold, preprocessTab.entry_cb, user_metadata_path, video_path, nav_path, self.laser_tracks, laser_label, laser_dist, start_sample, stop_sample, output_path)
        if result:
            messagebox.showinfo("Success", "Ecological profiler file has been written to {}".format(output_path))
        else:
            messagebox.showerror("Error", "Operation failed, please retry.")

class entryWindow(object):
    def __init__(self, master, title, message):
        self.top = tk.Toplevel(master)
        self.title = title
        self.label = ttk.Label(self.top, text=message)
        self.entry = ttk.Entry(self.top)
        self.entry.bind("<Return>", self.cleanup)
        self.buttonQuit = ttk.Button(self.top, text="Ok", command=self.cleanup)
        self.value = None
        self.label.pack(padx=10, pady=[10, 2])
        self.entry.pack(padx=10, pady=[2, 10], expand=True, fill="x")
        self.buttonQuit.pack(pady= 10)

    def cleanup(self, event=None):
        self.value = self.entry.get()
        self.top.destroy()

if __name__ == "__main__":
    app = Application()
    app.title("Benthic Video Toolbox")
    icon = tk.PhotoImage(data=icon_data)
    app.iconphoto(True, icon)
    app.mainloop()
