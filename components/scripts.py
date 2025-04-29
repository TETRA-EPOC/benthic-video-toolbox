import pathlib as pl
import pandas as pd
from datetime import timedelta, datetime
from bisect import bisect_right, bisect_left
import json
import subprocess as sp
from collections import defaultdict
from tkinter import messagebox
from tkinter import filedialog
import numpy as np

def test_time_format(time: str):
    if time.isnumeric():
        return True
    else:
        try:
            s = time.split(':')
            if len(s) == 3 and (x.isnumeric() for x in s):
                return True
            return False
        except ValueError:
            return False

def format_time(time: str):
    if time.isnumeric():
        ftime = timedelta(seconds=float(time))
    else:
        try:
            ftime = pd.to_timedelta(time)
        except ValueError as e:
            print("Error formatting time {}: {}".format(time, e))
            return
    return ftime

def create_metadata_file():
    return pd.DataFrame(index=['total'], columns=['source video file', 'cut video file', 'navigation file', 'biigle annotation file', 'START source video (absolute timestamp)', 'START cut time (relative)', 'END cut time (relative)', 
                                     'START cut time (absolute timestamp)', 'END cut time (absolute timestamp)', 'covered distance (m)',
                                    'average field width measured (m)', 'covered surface (m²)', 'contributors list'])

def read_cut_times_from_nav(navPath: str, metadata_path: str=None):
    p = pl.Path(navPath)
    if (not p.exists()):
        message = navPath + " does not exist. Please provide a valid file path."
        messagebox.showerror("Error", message)
        raise FileExistsError(message)
    elif (not p.is_file()):
        message = navPath + " is not a regular file. Please provide a valid file path."
        messagebox.showerror("Error", message)
        raise FileExistsError(message)
    try:
        df = pd.read_table(navPath, usecols=["HEURE", "CODEseq"])
    except ValueError as e:
        messagebox.showerror("Error", "Failed to read cut times from nav file: {}".format(e))
        return
    idx0 = df.index[df["CODEseq"] == "DEBPLO"]
    t0_string = df.at[idx0[0], "HEURE"]
    t0 = datetime.strptime(t0_string, "%H:%M:%S")
    idx1 = df.index[df["CODEseq"] == "FINFIL"]
    t1_string = df.at[idx1[0], "HEURE"]
    t1 = datetime.strptime(t1_string, "%H:%M:%S")
    idx2 = df.index[df["CODEseq"] == "DEBVIR"]
    t2_string = df.at[idx2[0], "HEURE"]
    t2 = datetime.strptime(t2_string, "%H:%M:%S")
    start = str(t1 - t0)
    end = str(t2 - t0)

    if metadata_path:
        mp = pl.Path(metadata_path)
        if (mp.suffix != ".csv"):       # extension is not null, user entered full path
            metadata_path = metadata_path.with_suffix(".csv")
        if mp.exists() and mp.is_file():
            metadata = pd.read_csv(mp, index_col=0)
        else:
            metadata = create_metadata_file()
        metadata.iat[0, 2] = navPath
        metadata.iat[0, 4] = t0_string
        metadata.iat[0, 5] = start
        metadata.iat[0, 6] = end
        metadata.iat[0, 7] = t1_string
        metadata.iat[0, 8] = t2_string
        metadata.to_csv(metadata_path)

    return [t0_string, start, end, t1_string, t2_string]

def cut_command(input, start, stop, output, metadata_path):
    p = pl.Path(metadata_path)
    if (p.suffix != ".csv"):       # extension is not null, user entered full path
        metadata_path = metadata_path.with_suffix(".csv")
    if p.exists() and p.is_file():
        metadata = pd.read_csv(p, index_col=0)
    else:
        metadata = create_metadata_file()
    metadata.iat[0, 0] = input
    metadata.iat[0, 1] = output
    # if start and stop don't match with nav file values, update user metadata field
    if pd.isnull(metadata.iat[0, 4]):
        fstart = format_time(start)
        fstop = format_time(stop)
        metadata.iat[0, 5] = fstart
        metadata.iat[0, 6] = fstop
    else:
        t0_abs = datetime.strptime(metadata.iat[0, 4], "%H:%M:%S")
        if start != metadata.iat[0, 5]:
            fstart = format_time(start)
            metadata.iat[0, 5] = fstart
            metadata.iat[0, 7] = datetime.strftime(t0_abs + fstart, "%H:%M:%S")
        if stop != metadata.iat[0, 6]:
            fstop = format_time(stop)
            metadata.iat[0, 6] = fstop
            metadata.iat[0, 8] = datetime.strftime(t0_abs + fstop, "%H:%M:%S")
    ffmpeg_cmd = "ffmpeg -loglevel error -i \"{}\" -ss {} -to {} -vcodec h264 -y \"{}\"".format(input, start, stop, output)
    ret = lambda : sp.Popen(ffmpeg_cmd, shell=True).wait()
    try:
        metadata.to_csv(metadata_path)
        return ret()
    except PermissionError as e:
        messagebox.showerror("Error", e)
        return 1
    except sp.TimeoutExpired:
        return ret()

def convert_nav_to_csv(
    navPath: str,
    videoName: str,
    callback,
    metadataPath: str,
    outPath: str = None,
    force: bool = False,
    volumeId: int = None,
    email: str = None,
    token: str = None): # -> pd.DataFrame:

    """Converts a Pagure's nav (.txt) file to a csv file readable by Biigle

    Python script that converts a navigation (.txt) file from a Pagure acquisition campain to a metadata (.csv) file readable by Biigle.

    Args:
        navPath (str): Full path to input navigation file (must be a .txt).
        videoName (str): Video filepath associated to this nav file. Required for writing in metadata.
        callback (function): a callback used to open an entry window where user can add a text input
        metadataPath (str): User metadata filepath used to keep track of all operations in BVT.
        outPath (str): Directory to write the output file. Optionnal, by default the file is saved in the same directory as input nav file.
        force (bool): Force overwrite if a .csv file with the same name is present in output directory.
        volumeId (int): Id of corresponding volume inside Biigle. If provided, the resulting file will be imported inside Biigle through API ids provided.
        email (str): User email used to connect to Biigle Rest API.
        token (str): User token used to connect to Biigle Rest API (see doc: https://calcul01.epoc.u-bordeaux.fr:8443/doc/api/index.html)
    """

    # Get directory and construct ouptput file name from input nav filename

    inputFilepath = pl.PurePath(navPath)
    if (not pl.Path(inputFilepath).exists()):
        message = navPath + "does not exist. Please provide a valid file path."
        messagebox.showerror("Error", message)
        raise FileNotFoundError(message)
    if not videoName:
        message = "Video name can not be empty"
        messagebox.showerror("Error", message)
        raise ValueError(message)
    if not pl.Path(videoName).suffix:
        videoName += ".mp4"

    # fill user metadata with nav file and source video filepath
    mp = pl.Path(metadataPath)
    if (mp.suffix != ".csv"):       # extension is not null, user entered full path
        metadataPath = metadataPath.with_suffix(".csv")
    if mp.exists() and mp.is_file():
        metadata = pd.read_csv(mp, index_col=0)
    else:
        metadata = create_metadata_file()
    # do not override video filepath if exists (extension may be wrong) but fill if empty, that means video has not been cut
    if pd.isnull(metadata.iat[0, 1]) and videoName != metadata.iat[0, 0]:
        metadata.iat[0, 1] = videoName
    if pd.isnull(metadata.iat[0, 2]) or metadata.iat[0, 2] != navPath:
        metadata.iat[0, 2] = navPath

    # Construct full output path
    if (pl.Path(outPath).suffix):       # extension is not null, user entered full path
        outFilepath = outPath
    else:
        inputFilename = inputFilepath.stem
        outFilename = inputFilename.replace("nav", "metadata")
        if (outFilename == inputFilename):
            message = "Navigation file should have 'nav' in filename, otherwise provide full output filename."
            messagebox.showerror("Error", message)
            raise Exception(message)

        if (pl.Path(outPath).is_dir()) :
            outFilepath = pl.PurePath(outPath).joinpath(outFilename + ".csv")
        else:
            outFilepath = inputFilepath.parent.joinpath(outFilename + ".csv")

    if (pl.Path(outFilepath).exists() and not force):
        message = outFilename + " already exists, if you want to overwrite it please use the --force parameter"
        messagebox.showerror("Error", message)
        raise FileExistsError(message)

    try:
        # indata = pd.read_table(navPath)
        df = pd.read_table(navPath, parse_dates={"taken_at" : ["DATE", "HEURE"]}, dayfirst=True)
    except ValueError as e:
        messagebox.showerror("Error", "Failed to read timestamp navigation file: {}".format(e))
        return
    nav_times = pd.to_timedelta(df.taken_at.astype(str).str.split(' ').str[-1])

    # get cut times and keep nav info accordingly
    if messagebox.askyesno(message="Was the video {} cut before being annotated ?".format(pl.Path(videoName).name)):
        if not metadata.iloc[0,4:9].isna().any():
            t0_abs, start_rel, stop_rel, start_cut_abs, stop_cut_abs = metadata.iloc[0,4:9]
        else:
            t0_abs, start_rel, stop_rel, start_cut_abs, stop_cut_abs = read_cut_times_from_nav(navPath)
        lines = ["The program found the following cut times (relative):", "{} and {} corresponding to absolute timestamps {} and {}".format(start_rel, stop_rel, start_cut_abs, stop_cut_abs), "Is it correct ?"]
        if not messagebox.askyesno(title="Use cut times ?", message="\n".join(lines)):
            if callback:
                start_offset = callback("Enter the 'start cut time' used to cut video (relative, hh:mm:ss or number of seconds):")
                stop_offset = callback("Enter the 'stop cut time' used to cut video (relative, hh:mm:ss or number of seconds):")
                if not start_offset or not stop_offset:     # user cancelled command
                    return
                try:
                    start_cut_abs = pd.to_timedelta(t0_abs) + format_time(start_offset)
                    stop_cut_abs = pd.to_timedelta(t0_abs) + format_time(stop_offset)
                except ValueError as e:
                    print("error: failed to convert string times {} and {} to timedelta".format(start_offset, stop_offset), e)
        else:
            try:
                start_cut_abs = pd.to_timedelta(start_cut_abs)
                stop_cut_abs = pd.to_timedelta(stop_cut_abs)
            except ValueError as e:
                    print("error: failed to convert string times {} and {} to timedelta".format(start_offset, stop_offset), e)
        if metadata.iloc[0,5:9].isna().any():
            # remove 'o days' added with timedelta conversion
            start_rel = str(start_rel).split(' ')[-1]
            stop_rel = str(stop_rel).split(' ')[-1]
            start_cut_abs_string = str(start_cut_abs).split(' ')[-1]
            stop_cut_abs_string = str(stop_cut_abs).split(' ')[-1]
            metadata.iloc[0,5:9] = start_rel, stop_rel, start_cut_abs_string, stop_cut_abs_string
    else:
        start_cut_abs = nav_times.iat[0]
        stop_cut_abs = nav_times.iat[-1]
    if pd.isnull(metadata.iat[0, 4]) or metadata.iat[0, 4] != t0_abs:
        metadata.iat[0, 4] = t0_abs
    i_start = max(0, bisect_left(nav_times, start_cut_abs))
    i_stop = min(bisect_left(nav_times, stop_cut_abs), len(nav_times)-1)

    df = df.iloc[i_start:i_stop+1]
    outdata = pd.DataFrame(columns=['file', 'taken_at'])
    outdata.taken_at = df.taken_at
    outdata.file = pl.Path(videoName).name
    if "LAT_PAGURE" and "LONG_PAGURE" in df.columns:
        latitudes = df.LAT_PAGURE
        longitudes = df.LONG_PAGURE
    else:
        lat_mask = df.columns.str.startswith(('lat', 'LAT', 'Lat'))
        lon_mask = df.columns.str.startswith(('lon', 'LON', 'Lon', 'lng'))
        # get first ids of masks (numpy arrays whith boolean values) meeting conditions
        lat_idx = np.where(lat_mask ==  True)[0][0]
        lon_idx = np.where(lon_mask ==  True)[0][0]
        latitudes = df.iloc[:,lat_idx]
        longitudes = df.iloc[:,lon_idx]
    if not latitudes.empty:
        outdata.insert(len(outdata.columns), "lat", latitudes)
    if not longitudes.empty:
        outdata.insert(len(outdata.columns), "lon", longitudes)
    yaw_mask = df.columns.str.startswith(('cap', 'CAP', 'yaw', 'YAW', 'heading', 'HEADING'))
    if yaw_mask.any():
        yaw_idx = np.where(yaw_mask == True)[0][0]
        outdata.insert(len(outdata.columns), "yaw", df.iloc[:,yaw_idx])

    try:
        outdata.to_csv(outFilepath, index=False)
        metadata.to_csv(metadataPath)
    except PermissionError as e:
        messagebox.showerror("Error", e)
        return

    if (volumeId):
        from components.biigle import Api
        from components.biigle import requests

        api = Api(email, token)

        with open(outFilepath, "rb") as f:
            try:
                api.post('volumes/{}/metadata'.format(volumeId), files={'file': f }, data={ 'parser': 'Biigle\Services\MetadataParsing\VideoCsvParser' })
            except requests.exceptions.RequestException as e:
                messagebox.showerror(title="Error: ", message=e)
                raise Exception(e)

    return True

def biigle_annot_to_yolo(
    csvPath: str,
    metadataPath: str,
    videoPaths: list = None,
    outPath: str = None):

    """Script to convert biigle's video annotations to yolo-formatted images annotations

    Python script that extracts frames annotations from a Biigle's video annotation csv report. Output is a folder containing one csv file for each time of each annotation.

    Args:
        csvPath (str): Full path to Biigle's CSV video annotations file
        metadataPath (str): User metadata filepath used to keep track of all operations in BVT.
        videoPath list(str): Full paths to input videos, used to extract frames with ffmpeg
        outPath (str): Output directory path where the yolo-formatted images annotations files will be saved
    """
    from components.utils import Track

    def convert_to_yolo_bbox(shape_id, coords, size_img):
        """Convert biigle annotation coordinates to YOLO bbox.

        :param shape_id (int): annotation shape identifier as set in biigle
        :param coords (tuple): annotation coordinates as set in biigle
        :param size_img (tuple): size of the image/frame (width_img, height_img)
        :return: a tuple of the YOLO bbox (x_center, y_center, width, height)
        """
        dw = 1.0 / size_img[0]
        dh = 1.0 / size_img[1]
        # shape_id = 7 means whole frame annotation, skip
        if shape_id == 7:
            return
        # for point annotation, we create a box with width and height of 3 pixels
        if shape_id == 1:
            x = max(min(coords[0], size_img[0]), 0) * dw
            y = max(min(coords[1], size_img[1]), 0) * dh
            w = 3.0 * dw
            h = 3.0 * dh
        else:
            # first we adjust all points coords so they are inside img (particular case for circle shape with radius as 3rd coord)
            if shape_id == 4:
                x_min = max(min(coords[0] - coords[2], size_img[0]), 0)
                x_max = max(min(coords[0] + coords[2], size_img[0]), 0)
                y_min = max(min(coords[1] - coords[2], size_img[1]), 0)
                y_max = max(min(coords[1] + coords[2], size_img[1]), 0)
            else:
                x_min = max(0, min(coords[::2]))
                x_max = min(size_img[0], max(coords[::2]))
                y_min = max(0, min(coords[1::2]))
                y_max = min(size_img[1], max(coords[1::2]))
            # then we create a rect bbox [x_center, y_center, w, h] with minimum surface that includes all points. In yolo format, coordinates are normalized between 0 and 1
            x = (x_min + x_max) / 2.0 * dw
            y = (y_min + y_max) / 2.0 * dh
            w = (x_max - x_min) * dw
            h = (y_max - y_min) * dh
        return (x, y, w, h)

    def read_data_from_csv(csvPath):
        """Get the annotation data from csv input file and fill dictionnnaries

        :param csvPath (str): Full path to Biigle's CSV video annotations file
        :return annotation_ids: 3D-dictionary dict(dict(list)) containing all annotation ids for each keytime of each video
                annotation_tracks: dictionary dict(list) mapping an annotation Track (or list of Track if there are gaps in annotation) to an annotation id
                contributors: dictionary dict(list) mapping a video filepath to list of contributors to this video annotations
        """
        rowdata = pd.read_csv(csvPath)
        classes = pd.DataFrame(columns=['class_name'])
        classes.index.name = 'class_id'
        annotation_ids = defaultdict(lambda: defaultdict(list))
        annotation_tracks = defaultdict(list)
        contrib = defaultdict(set)

        for row in rowdata.itertuples():
            video_filename = pl.Path(row.video_filename).stem       # retrieve video filename without the extension
            label_id = row.label_id
            label_name = row.label_name
            times = row.frames
            points = row.points
            video_annotation_label_id = row.video_annotation_label_id
            shape_id = row.shape_id
            try:
                attrs = row.attributes
            except AttributeError:
                message = "This method requires the 'attributes' column to be present in biigle annotation report. Please regenerate report if not (it was added from biigle reports module v4.29. All reports generated before july 2024 will not have it.)"
                messagebox.showerror(title="Error: ", message=message)
                raise AttributeError(message)
            attrs_dict = json.loads(attrs)
            width = attrs_dict["width"]
            height = attrs_dict["height"]
            track = Track(video_filename, label_id, shape_id, (width, height))
            classes.loc[label_id] = label_name
            contrib[video_filename].add(row.firstname + " " + row.lastname)

            # times and points are strings, convert them to arrays
            times = times[1:]
            times = times[:-1]
            times = times.split(",")    # array of strings: ['keyframe1', 'keyframe2', ...]
            # for points column, we need to remove two firsts and lasts character ('[[' and ']]') and split string with '],[' as delimiter to find the arrays corresponding to each time
            points = points[2:]
            points = points[:-2]
            points = points.split('],[')  # result is an array of strings: ['x11, x12, ...', 'x21, x22, ...' ...]
            for count, time in enumerate(times):
                coords = points[count]
                coords = coords.split(',')
                # fill a dictionnary with times as keys and annotation ids as values. We use try statement for time float conversion to handle gap case (null value) 
                try:
                    t = float(time)
                    track.add_keyframe(t, [float(coord) for coord in coords])
                    annotation_ids[video_filename][t].append(video_annotation_label_id)
                # time == null means there is a gap in annotation, save previous track and create a new one
                except:
                    annotation_tracks[video_annotation_label_id].append(track)
                    track = Track(video_filename, label_id, shape_id, (width, height))
            # add track to annotation_tracks dict
            annotation_tracks[video_annotation_label_id].append(track)

        classes.to_csv(pl.Path(outPath).joinpath("classes.txt"))
        return annotation_ids, annotation_tracks, contrib

    def process_keyframe(kf, annot_ids, tracks, memdata):
        '''Search for all annotations present at this keyframe and return YOLO-formatted data

        :param kf (float): current keyframe (time of the video)
        :param annot_ids (list(int)): list of the annotation_ids referencing this keyframe (there may be other annotation tracks present, we need to da a time check and interpolate position if so) 
        :return outdata: a pandas.Dataframe containing YOLO-formatted annotation data for this frame
        '''
        outdata = pd.DataFrame(columns=['class', 'x', 'y', 'w', 'h'])
        # outdata.index.name = 'video_annotation_label_id'

        # Browse all annotation ids referencing this keyframe to get corresponding tracks and update memdata with current tracks
        for i in annot_ids:
            try:
                # Look in annotation_tracks for track where keyframe is present (linked tracks don't overlap so the first found is ok)
                for track in tracks[i]:
                    if kf in track.keyframes:
                        coords = track.keyframes[kf]
                        break
            except ValueError:
                print("coords not found for annotation number {} at time {}".format(i, kf))
                continue

            bbox = convert_to_yolo_bbox(track.shape_id, coords, track.video_size)
            if not bbox:
                print("Empty bbox for annotation {}, track = {}".format(i, json.dumps(track)))
                continue

            outdata.loc[i] = {'time': kf, 'class': track.label_id, 'x': bbox[0], 'y': bbox[1], 'w': bbox[2], 'h': bbox[3]}
            if not i in memdata:
                memdata.add(i)
            if kf == max(track.keyframes.keys()):
                memdata.remove(i)

        # For all annotations that are in memdata (currently processed) but not in outdata (not referencing this keyframe) we interpolate its position with closest keyframes
        for j in memdata.difference(outdata.index):
            keys = []
            # Look in annotation_tracks for track where keyframe should be interpolated
            for track in tracks[j]:
                keys = list(track.keyframes.keys())
                if kf >= min(keys) and kf <= max(keys):
                    break
            if len(keys) == 0:
                print("couldn't find interpolation segment keyframes for annotation number {} at time {}".format(j, kf))
                continue
            # Find the two keyframe values t1 and t2 surrounding k in track.keyframes with bisect_right
            pos = bisect_right(keys, kf)
            if pos == 0 or pos == len(keys):
                raise ValueError("pos = {} and keys = {}".format(pos, keys))
            t1, t2 = keys[pos-1], keys[pos]
            # interpolate coordinates of annotation j at time k according to coordinates at t1 and t2
            c1, c2 = track.keyframes[t1], track.keyframes[t2]
            coeff = (kf - t1) / (t2 - t1)
            interpolated_coords = [c1[idx] + coeff * (c2[idx] - c1[idx]) for idx in range (min(len(c1), len(c2)))]
            bbox = convert_to_yolo_bbox(track.shape_id, interpolated_coords, track.video_size)
            if not bbox:
                print("Empty bbox for annotation {}, track = {}".format(i, json.dumps(track)))
                continue
            outdata.loc[j] = {'class': track.label_id, 'x': bbox[0], 'y': bbox[1], 'w': bbox[2], 'h': bbox[3]}
        return outdata, memdata

    def extract_annotated_frames(path, sorted_times, videoname, out_annotations_path, out_images_path):
        '''Extract from a video all frames with annotations and draw bboxes with opencv

        :param path (pathlib Path): Full path to input video file
        :param sorted_times (list(float)): List of annotation keytimes in ascending order
        :param out_annotations_path (pathlib Path): Full path to output annnotations files folder
        :param out_images_path (pathlib Path): Full path to output images files folder
        :return True if extraction went well, else False
        '''
        import cv2

        videoIn = cv2.VideoCapture(str(path))
        video_width = videoIn.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (video_width == 0 or video_height == 0 or not video_width or not video_height):
            message = "Invalid values for video width and height"
            messagebox.showerror("Error", message)
            return False
        try:
            for kf in sorted_times:
                videoIn.set(cv2.CAP_PROP_POS_MSEC, kf*1e3)
                ret, frame = videoIn.read()
                if ret:
                    # open yolo file to get bboxes coordinates
                    with open(pl.Path(out_annotations_path).joinpath(videoname + "_" + str(kf) + '.txt'), 'r') as file:
                        next(file)  # skip headers
                        for l in file:
                            data = l.split(",")
                            if len(data) < 5:
                                print("error can't find bbox coordinates in yolo file")
                                break

                            x_center_pixel = float(data[1]) * video_width
                            y_center_pixel = float(data[2]) * video_height
                            dw = float(data[3]) * video_width / 2.0
                            dh = float(data[4]) * video_height / 2.0
                            xmin = int(x_center_pixel - dw)
                            ymin = int(y_center_pixel - dh)
                            xmax = int(x_center_pixel + dw)
                            ymax = int(y_center_pixel + dh)
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                            cv2.imwrite(str(pl.Path(out_images_path).joinpath(path.stem + "_" + str(kf) + ".png")), frame)
        except Exception as error:
            messagebox.showerror("Error", "error processing video {}".format(videoname))
            return False
        # release resources
        videoIn.release()
        cv2.destroyAllWindows()
        return True

    # construct output paths for images and annotation files
    outImagesPath = pl.Path(outPath).joinpath("images")
    if not outImagesPath.exists():
        outImagesPath.mkdir()
    outAnnotationsPath = pl.Path(outPath).joinpath("labels")
    if not outAnnotationsPath.exists():
        outAnnotationsPath.mkdir()

    # fill user metadata with csv file, source video filepath (if only one source path) and contributors list
    mp = pl.Path(metadataPath)
    if (mp.suffix != ".csv"):       # extension is not null, user entered full path
        metadataPath = metadataPath.with_suffix(".csv")
    if mp.exists() and mp.is_file():
        metadata = pd.read_csv(mp, index_col=0)
    else:
        metadata = create_metadata_file()
    if pd.isnull(metadata.iat[0, 1]) and (len(videoPaths) == 1):
        metadata.iat[0, 1] = videoPaths[0]
    metadata.iat[0, 3] = csvPath

    annotation_ids, annotation_tracks, contributors = read_data_from_csv(csvPath)

    # For each video input file
    for videoname in annotation_ids.keys():
        # Keep track of annotations being processed in a set of their ids, to interpolate positions in-between key frames
        memdata = set()
        # Sort annotation_ids by keys (i.e. keyframes).
        sorted_ids = sorted(annotation_ids[videoname].items())

        # Browse all keyframes in ascending order
        for k, ids in sorted_ids:
            outdata, memdata = process_keyframe(k, ids, annotation_tracks, memdata)
            outFilepath = pl.Path(outAnnotationsPath).joinpath(videoname + "_" + str(k) + '.txt')
            if outdata.empty:
                print("error: data for timekey {} of file {} is empty".format(k, outFilepath))
                continue
            outdata.to_csv(outFilepath, index=False)

        if videoPaths:
            if (len(videoPaths) == 1):
                p = pl.Path(videoPaths[0])
                if (videoname != p.stem):
                    message = "videopath name {} doesn't match with name read in csv file {}".format(p.stem, videoname)
                    messagebox.showerror("Error", message)
                    continue
                else:
                    metadata.iat[0, 12] = str(contributors[videoname])
            else:
                for path in videoPaths:
                    p = pl.Path(path)
                    if videoname == p.stem:
                        break
            if not p:
                message = "Error can't determine video paths"
                messagebox.showerror("Error", message)
                continue

            if extract_annotated_frames(p, [s[0] for s in sorted_ids], videoname, outAnnotationsPath, outImagesPath):
                messagebox.showinfo(title="Success", message="YOLO-formatted data and annoted frames for video {} have been written to {}".format(videoname, outPath))
    try:
        metadata.to_csv(metadataPath)
    except PermissionError as e:
        messagebox.showerror("Error", e)
        return

def interpolate_nav_data(
    nav_days: pd.Series,
    nav_times: pd.Series,
    latitudes: pd.Series,
    longitudes: pd.Series,
    keytime: float,
    timeoffset: pd.Timedelta = None):
    """Takes info from Pagure nav file (timestamps, latitudes, longitudes) and a keytime in video and interpolates those data at this keytime.

    Python script that takes info from an input navigation file (saved as pandas Series), a keytime (corresponding to time of video where the annotation is closest to the lasers),
    and interpolates the absolute timestamp and 2 columns data (ex: GPS position (latitude + longitude)) of the annotation.

    Args:
        nav_days (pandas Series): List of the input data days read from nav file
        nav_times (pandas Series): List of the input data timestamps read from nav file
        latitudes (pandas Series): List of the input data latitudes read from nav file
        longitudes (pandas Series): List of the input data longitudes read from nav file
        keytime (float): Time in video where to interpolate the annotation timestamp and position
        timeoffset (pandas Timedelta): If the video has been cut before being annotated, we have to count timestamps from nav file with an offset
    """

    if timeoffset == timedelta(0) :
        time = timedelta(seconds=keytime)
    else:
        time = timeoffset + timedelta(seconds=keytime)

    # Find rightmost value less than or equal to time in nav_times
    i = min(max(bisect_right(nav_times, time), 1), len(nav_times)-1)
    if not i:
        raise ValueError
    # get time values wrapping time to compute interpolation coeff and latitude/longitude corresponding to these times
    t1, t2 = nav_times.iat[i-1], nav_times.iat[i]
    lat1, lat2 = latitudes.iat[i-1], latitudes.iat[i]
    lon1, lon2 = longitudes.iat[i-1], longitudes.iat[i]
    coeff = (time - t1) / (t2 - t1)
    time_delta = nav_days[i] + time
    lat_delta = lat1 + coeff * (lat2 - lat1)
    lon_delta = lon1 + coeff * (lon2 - lon1)
    return time_delta, lat_delta, lon_delta

def manual_detect_laserpoints(
        label: str,
        csvPath: str):
    """Get the laserpoints positions from a CSV video annotation file (output from Biigle)

    Python script that takes a CSV video annotation file from Biigle and the label used to annotate laserpoints in video, and returns a 2D-dictionnary dict(dict),
    mapping a tuple of laser tracks to a video filename: {'video name': (laser_track_1, laser_track_2), ...}. If only one video in CSV file, len(dict) = 1.
    Each laser track is a dict mapping a keyframe to a position in pixels in the image: {'time': coords}

    Args:
        label (str): Label used to annotate laserpoints in video inside biigle
        csvPath (str): Full path to input video annotation file (must be a .csv)
    """

    if (not pl.Path(csvPath).exists()):
        message = str(csvPath + "does not exist")
        messagebox.showerror("Error", message)
        raise FileNotFoundError(message)

    data = pd.read_csv(csvPath)
    laser_tracks = defaultdict(list)
    label_data = data[data["label_name"].str.fullmatch(label, case=False)]
    if label_data.empty:
        messagebox.showerror("Error", "Could not find label {} in annotation file. Please verify your data.".format(label))
    for row in label_data.itertuples():
        video_filename = row.video_filename
        track = {}
        times = row.frames
        points = row.points
        # times and points are strings, convert them to arrays
        times = times[1:]
        times = times[:-1]
        times = times.split(",")    # array of strings: ['keyframe1', 'keyframe2', ...]
        # for points column, we need to remove two firsts and lasts character ('[[' and ']]') and split string with '],[' as delimiter to find the arrays corresponding to each time
        points = points[2:]
        points = points[:-2]
        points = points.split('],[')  # result is an array of strings: ['x11, x12, ...', 'x21, x22, ...' ...]
        for count, time in enumerate(times):
            coords = points[count]
            coords = coords.split(',')
            # fill a dictionnary with times as keys and annotation ids as values. We use try statement for time float conversion to handle gap case (null value) 
            try:
                t = float(time)
                track.update({t: [float(coord) for coord in coords]})
            except:     # time == null means there is a gap in annotation, skip and continue on next keyframe
                continue
        laser_tracks[video_filename].append(track)
    return laser_tracks

def eco_profiler(
        csvPath: str,
        dy_max_str: float,
        callback,
        metadataPath: str,
        videoPath: str = None,
        navPath: str = None,
        laser_tracks: dict = None,
        laser_label : str = None,
        laser_dist_str: str = None,
        start_label: str = None,
        stop_label: str = None,
        outPath: str = None):

    """Build the ecological profiler file with various informations about annotated organisms, from the CSV video annotation file from biigle, input metadata (nav file) and computed laser postions.

    Python script that takes a CSV video annotation file from Biigle, a Pagure navigation file, laser tracks output from either manual or automatic detection and,
    for each annotation clip (i.e. annotations with at least 2 frames), find the frame where the annotation is closest to the lasers, compute and fill an output CSV file with:
        - video name
        - time offset i.e. the absolute timestamp at which the video starts (may differ from first time in nav file if video has been cut)
        for each annotation:
            - time of the video where the measure is done
            - absolute timestamp of the frame
            - the label and label hierarchy of this annotation
            - laser positions in the image (in pixels)
            - distance between lasers at that frame (in pixels)
            - annotation distance to lasers (in pixels)
            - position of the annotation in the image (in pixels)
            - computed GPS position of the organism at that timestamp
            - size of the annotation in the image
            - computed size of the organism

    Args:
        csvPath (str): Full path to input video annotation file (must be a .csv)
        dy_max_str (str): threshold distance to lasers line in y: annotations below are considered too far for the measure to be accurate (in % of video height)
        callback (function): a callback used to open an entry window where user can add a text input
        metadataPath (str): User metadata filepath used to keep track of all operations in BVT.
        videoPath (str): Video filepath associated to this nav file. Not mandatory, used if path not found in metadata file.
        navPath (str): Nav path to input navigation files (won't be considered if multiple videos in annotation file, overrided by user entries) (.txt or .csv)
        laser_tracks (dict): 2D-dict of lasers image positions mapped to keytime. Each pair of tracks are mapped to video filename
        laser_label (str): optionnal. Label used to annotate lasers if manual annotation was used.
        laser_dist_str (str): distance in cm between lasers (entered by user)
        str_timeoffset (str): optionnal. Time offset to start the timestamps count from nav file (str at format hh:mm:ss). If not filled in, read from nav file.
        start_label (str): Label used to delimit beginning of annotation sections (whole frame annotation label)
        stop_label (str): Label used to delimit ending of annotation sections (whole frame annotation label)
        outPath (str): Full path to write the output file. Optionnal, by default the input file is overwritten
    """
    import math

    def measure_annot_size(annot_coords, annot_shape_id):
        # point annotation, skip size measurement
        if annot_shape_id == 1:
            pass
        # for circles we take diameter (2 * third coordinate) as size
        elif annot_shape_id == 4:
            annot_size_px = 2*annot_coords[2]
        # for lines we take total length (sum of segment lengths) as size
        elif annot_shape_id == 2:
            annot_size_px = 0
            # zip coordinates as (x, y) pairs then re-zip as points pairs pt1=(x1, y1) and pt2=(x2, y2) to compute segment lengths
            coords_pts = list(zip(annot_coords[::2], annot_coords[1::2]))
            for pt1, pt2 in zip(coords_pts, coords_pts[1:]):
                annot_size_px += math.dist(pt1, pt2)
        # for rectangles we take diagonal as size. If it's a polygon with 4 points, assume it's close to parallelogram and take one diagonal
        elif annot_shape_id == 5 or len(annot_coords) == 8:
            pt1 = (annot_coords[0], annot_coords[1])
            pt2 = (annot_coords[4], annot_coords[5])
            annot_size_px = math.dist(pt1, pt2)
        else:
            messagebox.showerror("Error", "Annotation shape {} not supported, skipped.".format(annot_shape_id))
            pass
        return annot_size_px

    # function that takes nav_times and distances pd.Series from nav file and compute covered distance between start and stop cut times
    def compute_distance(nav_times, distances, start, stop):
        idx = min(max(bisect_left(nav_times, start), 0), len(nav_times)-1)
        stop_idx = min(max(bisect_left(nav_times, stop), 0), len(nav_times)-1)
        # time sample is included in one nav timestamp section
        if idx == stop_idx:
            t1, t2 = nav_times.iat[idx-1], nav_times.iat[idx]
            dist = (stop-start)/(t2-t1)*distances[idx]
        # time sample is larger than nav section timesteps, we have to sum distances of each step
        else:
            t1_start, t2_start = nav_times.iat[idx-1], nav_times.iat[idx]
            t1_stop, t2_stop = nav_times.iat[stop_idx-1], nav_times.iat[stop_idx]
            dist = (t2_start-start)/(t2_start-t1_start)*distances[idx]
            idx+=1
            while idx < stop_idx:
                dist += distances[idx]
                idx+=1
            dist += (stop-t1_stop)/(t2_stop-t1_stop)*distances[stop_idx]
        return dist

    if (not pl.Path(csvPath).exists()):
        message = csvPath + "does not exist"
        messagebox.showerror("Error", message)
        raise FileNotFoundError(message)
    p = pl.Path(outPath)
    if p.is_dir():
        if not p.exists():
            p.mkdir()
        outFilename = pl.PurePath(csvPath).stem
        outFilepath = p.joinpath(outFilename + "_eco_profiler.csv")
    else:
        if not p.parent.exists():
            p.parent.mkdir()
        outFilepath = outPath

    # fill user metadata with csv file, source video filepath (if only one source path) and contributors list
    mp = pl.Path(metadataPath)
    metadata = None
    if (mp.suffix != ".csv"):       # extension is not null, user entered full path
        metadataPath = metadataPath.with_suffix(".csv")
    if mp.exists() and mp.is_file():
        metadata = pd.read_csv(mp, index_col=0)
    else:
        metadata = create_metadata_file()
    if metadata.empty:
        messagebox.showerror("Error", "Failed to retrieve or create metadata file {}, exit.".format(metadataPath))
        return
    metadata.iat[0, 2] = navPath
    metadata.iat[0, 3] = csvPath
    # if video path is different from source path and cut video path not filled in metadata file, add it in 'cut video file' column
    if pd.isnull(metadata.iat[0, 1]) and videoPath != metadata.iat[0, 0]:
        metadata.iat[0, 1] = videoPath
    # try to get video filename from (in priority order): video path input, cut video path in metadata, source video path in metadata. If not found ask user.
    if videoPath:
        metadata_videoname = pl.Path(videoPath).name
    elif not pd.isnull(metadata.iat[0, 1]):
        metadata_videoname = pl.Path(metadata.iat[0, 1]).name
    elif not pd.isnull(metadata.iat[0, 0]):
        metadata_videoname = pl.Path(metadata.iat[0, 0]).name
    else:
        metadata_videoname = callback("Could not get video filename, please enter (cut) video filename with extension to be associated with this metadata file:")
        if not metadata_videoname:
            return

    data = pd.read_csv(csvPath)
    out_data = pd.DataFrame(index=data.index,
                            columns=['video_annotation_label_id', 'video_filename', 'label_name', 'label_hierarchy',
                                     'keytime (s)', 'timestamp', 'lasers_position_image (pixels)', 'lasers_distance_image (pixels)',
                                     'annotation_position_image (pixels)', 'annotation_GPS_position (lat, lon)', 'annotation_size_image (pixels)', 'annotation_size (cm)'])

    video_filenames = data["video_filename"].unique()
    for videoname in video_filenames:
        data_video = data[data["video_filename"] == videoname]
        first_row = next(data_video.itertuples())
        try:
            attrs = first_row.attributes
        except AttributeError:
            message = "This method requires the 'attributes' column to be present in biigle annotation report. Please regenerate report if not (it was added from biigle reports module v4.29. All reports generated before july 2024 will not have it.)"
            messagebox.showerror(title="Error: ", message=message)
            raise AttributeError(message)
        attrs_dict = json.loads(attrs)
        width = attrs_dict["width"]

        sample_start = []
        sample_stop = []
        if start_label and stop_label:
            start_data = data_video[data_video["label_name"].str.fullmatch(start_label, case=False)]
            stop_data = data_video[data_video["label_name"].str.fullmatch(stop_label, case=False)]
            if start_data.empty or stop_data.empty:
                messagebox.showerror("Error", "Could not find sample labels in annotation file, please verify your data.")
                return
            for row_start in start_data.itertuples():
                time_str = row_start.frames
                # whole frame annotations' frames values are strings: '[t]' (one time value in brackets)
                time_str = time_str[1:]
                time_str = time_str[:-1]
                try:
                    sample_start.append(float(time_str))
                except ValueError as e:
                    print("Error processing start marker annotations: ", e)
            for row_stop in stop_data.itertuples():
                time_str = row_stop.frames
                time_str = time_str[1:]
                time_str = time_str[:-1]
                try:
                    sample_stop.append(float(time_str))
                except ValueError as e:
                    print("Error processing stop marker annotations: ", e)
            sample_start = sorted(sample_start)
            sample_stop = sorted(sample_stop)
            if len(sample_start) != len(sample_stop):
                messagebox.showerror("Error", "Found different numbers of start and stop markers for video {}, please verify your data.".format(videoname))
                return
            elif len(sample_start) == 0 or len(sample_stop) == 0:
                messagebox.showerror("Error", "Could not compute one of samples track for video {}, please verify your data.".format(videoname))
                return
            # if laser tracks detected
            if laser_tracks:
                tracks = laser_tracks[videoname]
                # try to reorder them in couples with sample markers, only in full mode or if this laser data was never processed. Each sample should have one pair of lasers associated.
                if len(tracks) != len(sample_start):
                    laser_pairs = defaultdict(list)
                    for track in tracks:
                        t_start = next(iter(track.keys()))
                        pos_start = bisect_left(sample_start, t_start)
                        pos_stop = min(bisect_left(sample_stop, t_start), len(sample_stop)-1)
                        if pos_start != pos_stop:
                            print("Error: found different samples position ({} and {}) for laser track: t_start={}".format(pos_start, pos_stop, t_start))
                            continue
                        if len(laser_pairs[pos_start]) == 2:
                            print("Error: already found 2 lasers at pos_start {}".format(pos_start))
                            continue
                        laser_pairs[pos_start].append(track)
                    laser_tracks[videoname] = laser_pairs
        if len(video_filenames) > 1 or not navPath:
            nav_path = filedialog.askopenfilename(title="Select navigation file for input video {}".format(videoname), filetypes=[('all', '*'), ('text files', '*.txt'), ('csv files', '*.csv')])
        else:
            nav_path = navPath
        laser_dist = float(laser_dist_str) if laser_dist_str else None
        data_nav = pd.read_table(nav_path)
        nav_days = pd.to_datetime(data_nav["DATE"], dayfirst=True)
        nav_times = pd.to_timedelta(data_nav["HEURE"])
        if "LAT_PAGURE" and "LONG_PAGURE" in data_nav.columns:
            latitudes = data_nav["LAT_PAGURE"]
            longitudes = data_nav["LONG_PAGURE"]
        else:
            lat_mask = data_nav.columns.str.startswith(('lat', 'LAT', 'Lat'))
            lon_mask = data_nav.columns.str.startswith(('lon', 'LON', 'Lon', 'lng'))
            # get first ids of masks (numpy arrays whith boolean values) meeting regex conditions
            lat_idx = np.where(lat_mask == True)[0][0]
            lon_idx = np.where(lon_mask == True)[0][0]
            latitudes = data_nav.iloc[:,lat_idx]
            longitudes = data_nav.iloc[:,lon_idx]

        t0_abs = nav_times.iat[0]
        start_cut_abs, stop_cut_abs, start_rel, stop_rel = None, None, None, None
        if messagebox.askyesno(message="Was the video {} cut before being annotated ?".format(videoname)):
            if metadata_videoname == videoname and not metadata.iloc[0,4:9].isna().any():
                t0_abs, start_rel, stop_rel, start_cut_abs, stop_cut_abs = metadata.iloc[0,4:9]
            else:
                t0_abs, start_rel, stop_rel, start_cut_abs, stop_cut_abs = read_cut_times_from_nav(nav_path)
            lines = ["The program found the following 'start cut time':", "{} (relative time) corresponding to {} absolute timestamp".format(start_rel, start_cut_abs), "Is it correct ?"]
            if not messagebox.askyesno(title="Use cut times ?", message="\n".join(lines)):
                if callback:
                    start_offset = callback("Enter the 'start cut time' used to cut video (relative, hh:mm:ss or number of seconds):")
                    stop_offset = callback("Enter the 'stop cut time' used to cut video (relative, hh:mm:ss or number of seconds):")
                    if not start_offset or not stop_offset:     # user cancelled command
                        return
                    try:
                        start_rel = format_time(start_offset)
                        stop_rel = format_time(stop_offset)
                        start_cut_abs = pd.to_timedelta(t0_abs) + start_rel
                        stop_cut_abs = pd.to_timedelta(t0_abs) + stop_rel
                    except ValueError as e:
                        print("error: failed to convert string times {} and {} to timedelta".format(start_offset, stop_offset), e)
            else:
                try:
                    start_cut_abs = pd.to_timedelta(start_cut_abs)
                    stop_cut_abs = pd.to_timedelta(stop_cut_abs)
                except ValueError as e:
                    print("error: failed to convert string times {} and {} to timedelta".format(start_offset, stop_offset), e)
            # if cut times out of bounds
            if start_cut_abs < nav_times.iat[0]:
                start_cut_abs = nav_times.iat[0]
                start_rel = timedelta(0)
            if stop_cut_abs > nav_times.iat[-1]:
                stop_cut_abs = nav_times.iat[-1]
                stop_rel = nav_times.iat[-1] - nav_times.iat[0]
            # update eventually metadata file
            if metadata_videoname == videoname and metadata.iloc[0,5:9].isna().any():
                # remove 'o days' added with timedelta conversion
                start_rel = str(start_rel).split(' ')[-1]
                stop_rel = str(stop_rel).split(' ')[-1]
                start_cut_abs_string = str(start_cut_abs).split(' ')[-1]
                stop_cut_abs_string = str(stop_cut_abs).split(' ')[-1]
                metadata.iloc[0,5:9] = start_rel, stop_rel, start_cut_abs_string, stop_cut_abs_string
        else:
            start_cut_abs = nav_times.iat[0]
            stop_cut_abs = nav_times.iat[-1]
        if metadata_videoname == videoname:
            if metadata.iat[0,4] != t0_abs:
                metadata.iat[0,4] = t0_abs

            distances = []
            dist_mask = data_nav.columns.str.startswith(('Dist', 'dist', 'DIST'))
            dist_idx = np.where(dist_mask == True)[0][0]
            distances = pd.to_numeric(data_nav.iloc[:, dist_idx])
            if len(distances) > 0 and start_cut_abs and stop_cut_abs:
                total_distance = compute_distance(nav_times, distances, start_cut_abs, stop_cut_abs)
                metadata.iat[0, 9] = total_distance
                average_field_width = 0
                if len(sample_start) > 0 and len(sample_stop) > 0:
                    # compute covered distance for each sample
                    for idx in range(len(sample_start)):
                        start_sample = timedelta(seconds=sample_start[idx])
                        stop_sample = timedelta(seconds=sample_stop[idx])
                        start_sample_abs = start_cut_abs + start_sample
                        stop_sample_abs = start_cut_abs + stop_sample
                        dist = compute_distance(nav_times, distances, start_sample_abs, stop_sample_abs)
                        new_row = pd.Series([start_sample, stop_sample, start_sample_abs, stop_sample_abs, dist], index=metadata.columns[5:10], name="sample {}".format(idx))
                        new_row = new_row.astype(str).str.split(' ').str[-1]
                        if not "sample {}".format(idx) in metadata.index:
                            metadata = pd.concat([metadata, new_row.to_frame().T])
                        else:
                            metadata.loc["sample {}".format(idx), metadata.columns[5:10]] = new_row

                        # if laser tracks detected compute average field widths for each sample
                        if laser_tracks and idx in laser_tracks[videoname]:
                            sample_field_width = 0
                            pair_tracks = laser_tracks[videoname][idx]
                            if len(pair_tracks) < 2:
                                print("Error: can't find laser couple tracks for annotation {}. sample_start[{}] = {})".format(video_annotation_label_id, idx, sample_start[idx]))
                            else:
                                track_l1 = pair_tracks[0]
                                track_l2 = pair_tracks[1]
                            kfs_l2 = list(track_l2.keys())
                            for k, icoords_l1 in track_l1.items():
                                # laser annotation should be point or circle, keep only 2 first coordinates x,y
                                icoords_l1 = icoords_l1[0:2]
                                if k in track_l2:
                                    icoords_l2 = track_l2[k][0:2]
                                else:
                                    pos = min(bisect_right(kfs_l2, k), len(kfs_l2)-1)
                                    tmin, tmax = kfs_l2[pos-1], kfs_l2[pos]
                                    # interpolate coordinates of lasers l2 at time k according to coordinates at tmin and tmax
                                    cmin, cmax = track_l2[tmin], track_l2[tmax]
                                    coeff = (k - tmin) / (tmax - tmin)
                                    icoords_l2 = (cmin[0] + coeff * (cmax[0] - cmin[0]), cmin[1] + coeff * (cmax[1] - cmin[1]))
                                dist_lasers_px = math.dist(icoords_l1, icoords_l2)
                                if dist_lasers_px > 0:
                                    sample_field_width += 1/dist_lasers_px
                            # laser_dist is in cm, we want field_width in m
                            sample_field_width = sample_field_width * laser_dist * width / float(len(track_l1)*100)
                            average_field_width += sample_field_width
                            if "sample {}".format(idx) in metadata.index:
                                metadata.loc["sample {}".format(idx), metadata.columns[10]] = sample_field_width
                                metadata.loc["sample {}".format(idx), metadata.columns[11]] = dist * sample_field_width
                    average_field_width /= len(sample_start)
                    metadata.iat[0, 10] = average_field_width
                    metadata.iat[0, 11] = total_distance * average_field_width
                # compute average field width by looping through laser tracks
                elif laser_tracks:
                    track_l1 = laser_tracks[videoname][0]
                    track_l2 = laser_tracks[videoname][1]
                    kfs_l2 = list(track_l2.keys())
                    for k, icoords_l1 in track_l1.items():
                        icoords_l1 = icoords_l1[0:2]
                        if k in track_l2:
                            icoords_l2 = track_l2[k][0:2]
                        else:
                            pos = min(bisect_right(kfs_l2, k), len(kfs_l2)-1)
                            tmin, tmax = kfs_l2[pos-1], kfs_l2[pos]
                            # interpolate coordinates of lasers l2 at time k according to coordinates at tmin and tmax
                            cmin, cmax = track_l2[tmin], track_l2[tmax]
                            coeff = (k - tmin) / (tmax - tmin)
                            icoords_l2 = (cmin[0] + coeff * (cmax[0] - cmin[0]), cmin[1] + coeff * (cmax[1] - cmin[1]))
                        dist_lasers_px = math.dist(icoords_l1, icoords_l2)
                        if dist_lasers_px > 0:
                            average_field_width += 1/dist_lasers_px
                    # laser_dist is in cm, we want field_width in m
                    average_field_width = average_field_width * laser_dist * width / float(len(track_l1) * 100)
                    metadata.iat[0, 10] = average_field_width
                    metadata.iat[0, 11] = total_distance * average_field_width

        contributors = set()
        for row in data_video.itertuples():
            video_annotation_label_id = row.video_annotation_label_id
            label_name = row.label_name
            shape_id = row.shape_id
            times = row.frames
            points = row.points
            # times and points are strings, convert them to arrays
            times = times[1:]
            times = times[:-1]
            times = times.split(",")    # array of strings: ['keyframe1', 'keyframe2', ...]
            # for points column, we need to remove two firsts and lasts character ('[[' and ']]') and split string with '],[' as delimiter to find the arrays corresponding to each time
            points = points[2:]
            points = points[:-2]
            points = points.split('],[')  # result is an array of strings: ['x11, x12, ...', 'x21, x22, ...' ...]
            if laser_label and label_name.lower() == laser_label.lower():    # laser annotation
                continue
            try:
                attrs = row.attributes
            except AttributeError:
                message = "This method requires the 'attributes' column to be present in biigle annotation report. Please regenerate report if not (it was added from biigle reports module v4.29. All reports generated before july 2024 will not have it.)"
                messagebox.showerror(title="Error: ", message=message)
                raise AttributeError(message)
            attrs_dict = json.loads(attrs)
            height = attrs_dict["height"]
            dy_max = float(dy_max_str) * 0.01 * height
            if metadata_videoname == videoname:
                contributors.add(row.firstname + " " + row.lastname)

            dist_min = None
            t_min = None
            l1_min, l2_min = None, None
            l1_pos, l2_pos = None, None
            c_min = None
            coords_min = None
            laser_dist_px = None
            annot_size_px = None
            annot_size = None
            if not laser_tracks or len(times) == 1:
                # if no laser tracks, take the last keytime of annotation to get timestamp and GPS position, as we assume it is the closest position to camera, where measures are made
                t_min = float(times[-1])
                if shape_id != 7:     # not whole frame annotation
                    coords = points[-1]
                    coords = coords.split(',')
                    coords = [float(c) for c in coords]
                    n = float(len(coords)/2)
                    # except for points or circles, compute the barycenter of the annotation
                    if shape_id != 1 and shape_id != 4:
                        c_min = (sum(coords[::2]) / n, sum(coords[1::2]) / n)
                    else:
                        c_min = (coords[0], coords[1])
            else:
                track_l1, track_l2 = None, None
                # search for tracks in laser_tracks corresponding to this video
                if not videoname in laser_tracks:
                    message = "No laser annotation for video filename {}. Please verify your data.".format(videoname)
                    messagebox.showerror(title="Error: ", message=message)
                    continue
                if len(sample_start) > 0:
                    # search for correct section for this annotation with first timekey and get corresponding laser tracks
                    t_0 = float(times[0])
                    pos = max(0, bisect_right(sample_start, t_0) - 1)
                    pair_tracks = laser_tracks[videoname][pos]
                    if len(pair_tracks) < 2:
                        print("Error: can't find laser pair tracks for annotation {}. t_0 = {} and sample_start pos found at {}".format(video_annotation_label_id, t_0, sample_start[pos]))
                    else:
                        track_l1 = pair_tracks[0]
                        track_l2 = pair_tracks[1]
                # if annotations in "full" mode, jsut take first and second tracks from laser_tracks
                else:
                    # if more than 2 laser tracks found in full mode, print error message and return
                    if len(laser_tracks[videoname]) > 2:
                        message = "More than 2 lasers found for video filename {}. Please verify your data.".format(videoname)
                        messagebox.showerror(title='Error', message=message)
                        return
                    track_l1 = laser_tracks[videoname][0]
                    track_l2 = laser_tracks[videoname][1]
                if not track_l1 or not track_l2:
                    message = "At least one laser track is empty for annotation {}. Please verify your data.".format(video_annotation_label_id)
                    print(message)
                    continue

                # Look in laser tracks for segment-keyframe where annotation should be interpolated with bisect-right method
                kfs_l1 = list(track_l1.keys())
                kfs_l2 = list(track_l2.keys())
                for count, time in enumerate(times):
                    coords = points[count]
                    coords = coords.split(',')
                    # fill a dictionnary with times as keys and annotation ids as values. We use try statement for time float conversion to handle gap case (null value)
                    if time == "null":
                        # time == null means there is a gap in annotation, skip and continue on next keyframe
                        continue
                    try:
                        t = float(time)
                        # if annotation keytime is out of lasers bounds, take extreme values
                        if t < kfs_l1[0] or t < kfs_l2[0]:
                            icoords_l1 = track_l1[kfs_l1[0]]
                            icoords_l2 = track_l2[kfs_l2[0]]
                        elif t >= kfs_l1[-1] or t >= kfs_l2[-1]:
                            icoords_l1 = track_l1[kfs_l1[-1]]
                            icoords_l2 = track_l2[kfs_l2[-1]]
                        else:
                            # Find the two keyframe pairs tmin and tmax surrounding t in laser keyframes with bisect_right
                            pos_l1 = min(bisect_right(kfs_l1, t), len(kfs_l1)-1)
                            pos_l2 = min(bisect_right(kfs_l2, t), len(kfs_l2)-1)
                            tmin_l1, tmax_l1 = kfs_l1[pos_l1-1], kfs_l1[pos_l1]
                            tmin_l2, tmax_l2 = kfs_l2[pos_l2-1], kfs_l2[pos_l2]
                            # interpolate coordinates of lasers l1 and l2 at time t according to coordinates at tmin and tmax
                            cmin_l1, cmax_l1 = track_l1[tmin_l1], track_l1[tmax_l1]
                            coeff_l1 = (t - tmin_l1) / (tmax_l1 - tmin_l1)
                            icoords_l1 = (cmin_l1[0] + coeff_l1 * (cmax_l1[0] - cmin_l1[0]), cmin_l1[1] + coeff_l1 * (cmax_l1[1] - cmin_l1[1]))
                            cmin_l2, cmax_l2 = track_l2[tmin_l2], track_l2[tmax_l2]
                            coeff_l2 = (t - tmin_l2) / (tmax_l2 - tmin_l2)
                            icoords_l2 = (cmin_l2[0] + coeff_l2 * (cmax_l2[0] - cmin_l2[0]), cmin_l2[1] + coeff_l2 * (cmax_l2[1] - cmin_l2[1]))
                        # compute the equation of the line passing through both interpolated coords
                        if icoords_l2[0] - icoords_l1[0] == 0:
                            print("Error: 2 laser coords mixed, this souldn't happen.")
                            continue
                        a = (icoords_l2[1] - icoords_l1[1]) / (icoords_l2[0] - icoords_l1[0])
                        b = icoords_l1[1] - a * icoords_l1[0]
                        # convert string coords list to float and compute barycenter of the polygon annotation (except for points and circles)
                        coords = [float(coord) for coord in coords]
                        n = float(len(coords)/2)
                        # except for points or circles, compute the barycenter of the annotation
                        if shape_id != 1 and shape_id != 4:
                            barycenter = (sum(coords[::2]) / n, sum(coords[1::2]) / n)
                        else:
                            barycenter = (coords[0], coords[1])
                        #  compute distance to laser line
                        dist = abs(barycenter[1] - a * barycenter[0] - b) / math.sqrt(1 + math.pow(a,2))
                        if not dist_min or dist < dist_min:
                            dist_min = dist
                            t_min = t
                            c_min = barycenter
                            l1_min, l2_min = icoords_l1, icoords_l2
                            coords_min = coords
                    except:
                        print("Error: couldn't find interpolation segment at {} for video {}, annotation {}: ".format(t, videoname, video_annotation_label_id))
                        continue
                if l1_min and l2_min:
                    laser_dist_px = math.dist(l1_min, l2_min)
                    l1_pos = (round(l1_min[0], 2), round(l1_min[1], 2))
                    l2_pos = (round(l2_min[0], 2), round(l2_min[1], 2))
                # below dy_max annotation is considered too far from lasers: do not compute size measurement
                if dist_min and (dist_min < dy_max):
                    annot_size_px = measure_annot_size(coords_min, shape_id)
                    if laser_dist and laser_dist_px and annot_size_px:
                        annot_size = annot_size_px * laser_dist/laser_dist_px

            out_data.at[row.Index, "video_annotation_label_id"] = video_annotation_label_id
            out_data.at[row.Index, "video_filename"] = videoname
            out_data.at[row.Index, "label_name"] = label_name
            if row.label_hierarchy:
                out_data.at[row.Index, "label_hierarchy"] = row.label_hierarchy
            if t_min != None:
                out_data.at[row.Index, "keytime (s)"] = round(t_min, 2)
                try:
                    timestamp, latitude, longitude = interpolate_nav_data(nav_days, nav_times, latitudes, longitudes, t_min, start_cut_abs)
                    out_data.at[row.Index, "timestamp"] = timestamp
                    out_data.at[row.Index, "annotation_GPS_position (lat, lon)"] = latitude, longitude
                except ValueError as e:
                    print("Error: failed to interpolate nav data at {}, skip.".format(t_min), e)
            else:
                print("could not find t_min for annotation {}".format(video_annotation_label_id))
            if l1_pos and l2_pos:
                out_data.at[row.Index, "lasers_position_image (pixels)"] = l1_pos, l2_pos
            if laser_dist_px:
                out_data.at[row.Index, "lasers_distance_image (pixels)"] = round(laser_dist_px, 2)
            if c_min:
                out_data.at[row.Index, "annotation_position_image (pixels)"] = (round(c_min[0], 2), round(c_min[1], 2))
            if annot_size_px:
                out_data.at[row.Index, "annotation_size_image (pixels)"] = annot_size_px
            if annot_size:
                out_data.at[row.Index, "annotation_size (cm)"] = annot_size

        if metadata_videoname == videoname:
            metadata.iat[0, 12] = str(contributors)
    try:
        metadata.to_csv(metadataPath)
        out_data.to_csv(outFilepath, index=False)
    except PermissionError as e:
        messagebox.showerror("Error could not write output file: ", e)
        return False
    return True
