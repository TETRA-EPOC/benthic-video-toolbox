import sys
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
    elif(platform.system() == "Windows" and wslPath.is_posix_path(path)):
        path = wslPath.to_windows(path)
    return path

def get_video_path(page, event=None):
    page.video_path = convert_path(page.video_entry.get())
    page.video_entry.delete(0, "end")
    page.video_entry.insert('end', page.video_path)

def get_nav_path(page, event=None):
    page.nav_path = convert_path(page.nav_entry.get())
    page.nav_entry.delete(0, "end")
    page.nav_entry.insert('end', page.nav_path)

def get_csv_path(page, event=None):
    page.csv_path = convert_path(page.csv_entry.get())
    page.csv_entry.delete(0, "end")
    page.csv_entry.insert('end', page.csv_path)

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
        self.video_entry_frame = ttk.Frame(self)
        self.video_entry_frame.pack(padx=10, pady=10, fill="x")
        self.video_entry_label = ttk.Label(self.video_entry_frame, text="Video file path:")
        self.video_entry_label.pack(side="left", padx=2)
        if 'tkinterDnD' in sys.modules:
            self.video_entry = ttk.Entry(self.video_entry_frame, ondrop=lambda event: drop(self.video_string, event), text=self.video_string)
        else:
            self.video_entry = ttk.Entry(self.video_entry_frame, text=self.video_string)
        self.video_entry.pack(fill="x", padx=2)
        if 'wslPath' in sys.modules:
            self.video_entry.bind("<FocusOut>", lambda event: get_video_path(self))

        self.nav_string = tk.StringVar()
        self.nav_entry_frame = ttk.Frame(self)
        self.nav_entry_frame.pack(fill="x", padx=10, pady=10) #expand=True,
        self.nav_entry_label = ttk.Label(self.nav_entry_frame, text="Nav. file path:")
        self.nav_entry_label.pack(side="left", padx=2)
        if 'tkinterDnD' in sys.modules:
            self.nav_entry = ttk.Entry(self.nav_entry_frame, ondrop=lambda event: drop(self.nav_string, event), text=self.nav_string)
        else:
            self.nav_entry = ttk.Entry(self.nav_entry_frame, text=self.nav_string)
        self.nav_entry.pack(fill="x", padx=2)
        if wslPath in sys.modules:
            self.nav_entry.bind("<FocusOut>", lambda event: get_nav_path(self))

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
        self.load_frame.pack(anchor="e", padx=10, pady=[10, 20])
        self.load_label = ttk.Label(self.load_frame, text="Load metadata file into Biigle with volume id:")
        self.load_label.pack(side="left", padx=2)
        self.volume_id_entry = ttk.Entry(self.load_frame, width=10)
        self.volume_id_entry.pack(padx=20)

    def remove_example_cb(self, entry, event=None):
        if str(entry.cget("foreground")) == "#A9A9A9":
            entry.delete(0, "end")
            entry.configure(foreground="black")

    def reset_example_cb(self, entry, text, event=None):
        if not entry.get():
            entry.insert(0, text)
            entry.configure(foreground="#A9A9A9")

    def cut_video(self):
        self.video_path = self.video_entry.get()
        self.nav_path = self.nav_entry.get()
        if (not self.video_path):
            self.video_path = filedialog.askopenfilename(parent=self, title="Choose a video file to cut", filetypes=[('all', '*'), ('avi videos', '*.avi'), ('mp4 videos', '*.mp4')])
        p = pl.Path(self.video_path)
        if (not p.exists()):
            raise FileExistsError(p + " does not exist. Please provide a valid file path.")
        elif (not p.is_file()):
            raise FileExistsError(p + " is not a regular file. Please provide a valid file path.")
        if (not self.nav_path or len(self.nav_path) == 0):
            if (str(self.cut_to_entry.cget("foreground")) or str(self.cut_from_entry.cget("foreground"))) == "#A9A9A9" or not self.cut_from_entry.get() or not self.cut_to_entry.get():
                messagebox.showerror(title="Error: ", message="If no navigation file is provided, you need to specify 'From' and 'To' times (with format HH:MM:SS or in seconds) to cut video file.")
                return
            else:
                start = str(self.cut_from_entry.get())
                end = str(self.cut_to_entry.get())
                if not scripts.test_time_format(start) or not scripts.test_time_format(end):
                    messagebox.showerror(title="Error: ", message="Wrong format for start/end values. Please use seconds or HH:MM:SS.")
                    return
        else:
            start, end = scripts.read_times_from_nav(self.nav_path)
            lines = ["The program found the following cutting times in navigation file:", "{} and {}".format(start, end), "Do you want to proceed ?"]
            if not (messagebox.askokcancel(title="Confirm", message="\n".join(lines))):
                return
            self.cut_from_entry.delete(0, "end")
            self.cut_from_entry.insert(0, start)
            self.cut_to_entry.delete(0, "end")
            self.cut_to_entry.insert(0, end)
        output_path = filedialog.asksaveasfilename(parent=self, title="Save as", filetypes=[('mp4 videos', '*.mp4'), ('avi videos', '*.avi'), ('mpeg videos', '*.mpeg'),
                                                    ('quicktime videos', '*.mov'), ('all files', '*')], defaultextension='.mp4')
        result = scripts.cut_command(self.video_path, start, end, output_path)
        if result == 0:
            messagebox.showinfo("Success", "Video {} has been successfully cut and has been saved to {}".format(p.name, output_path))
        else:
            messagebox.showerror("Error", "Operation failed, please retry.")

    def convert_nav_to_csv(self):
        self.video_path = self.video_string.get()
        self.nav_path = self.nav_entry.get()
        if not self.nav_path:
            self.nav_path = filedialog.askopenfilename(title="Choose a Pagure navigation file to convert", filetypes=[('text files', '*.txt')])
            if not self.nav_path:   # if user cancelled command
                return
        apiTab = app.tabControl.nametowidget(app.tabControl.tabs()[2])
        email = apiTab.email_entry.get()
        token = apiTab.token_entry.get()
        volume_id = self.volume_id_entry.get()
        if self.volume_id_entry.get() and (not email or not token):
            messagebox.showerror(title="Error: ", message="To connect to Biigle API, please fill in the login details inside 'Biigle API Login' tab.")
            return
        if not self.video_path:
            window = videonameWindow(self)
            self.wait_window(window.top)
            video_name = window.value
            if not video_name:   # if user cancelled command
                return
        else:
            video_name = pl.Path(self.video_path).name
        output_path = filedialog.asksaveasfilename(parent=self, title="Save as", initialdir=pl.Path(self.nav_path).parent, filetypes=[('csv files', '*.csv'), ('all files', '*')], defaultextension='.csv')
        if not output_path:
            return
        result = scripts.convert_nav_to_csv(self.nav_path, video_name, output_path, True, volume_id, email, token)
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
        self.csv_entry_label = ttk.Label(self.csv_entry_frame, text="Biigle video annotation file path:")
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
        self.biigle_to_yolo_button.pack(padx=20, pady=10)

    def biigle_to_yolo(self):
        self.csv_path = self.csv_entry.get()
        preprocessTab = app.tabControl.nametowidget(app.tabControl.tabs()[0])
        if preprocessTab.video_entry.get():
            video_paths = [preprocessTab.video_entry.get()]
        else:
            video_paths = filedialog.askopenfilenames(title="Select the input video file(s) on which the annotations had been processed", filetypes=[('mp4 files', '*.mp4'), ('avi files', '*.avi')])
        output_path = filedialog.askdirectory(parent=self, title="Save as", mustexist=True)
        if not output_path:
            messagebox.showerror(title="Error", message="Conversion failed, please retry.")
            return
        scripts.biigle_annot_to_yolo(self.csv_path, video_paths, output_path)

    def add_geocoords(self):
        self.csv_path = self.csv_entry.get()
        if not self.csv_path:
            self.csv_path = filedialog.askopenfilename(title="Choose a CSV video annotation file", filetypes=[("csv files", "*.csv")])
        preprocessTab = app.tabControl.nametowidget(app.tabControl.tabs()[0])
        nav_path = preprocessTab.nav_entry.get()
        if not nav_path:
            nav_path = filedialog.askopenfilename(title="Choose a Pagure navigation file linked to this dataset", filetypes=[('text files', '*.txt')])
        output_path = filedialog.asksaveasfilename(parent=self, title="Save as", initialdir=pl.Path(self.csv_path).parent, filetypes=[("csv files", "*.csv"), ('text files', '*.txt')])
        if not output_path or not nav_path:
            messagebox.showerror(title="Error", message="Operation failed, please retry.")
            return
        scripts.add_geocoords(self.csv_path, nav_path, output_path)

class videonameWindow(object):
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.title=("Video name")
        self.label = ttk.Label(self.top, text="Enter the video filename associated with this navigation file:")
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
