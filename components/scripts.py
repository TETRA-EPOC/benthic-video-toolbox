import pathlib as pl
import pandas as pd
from datetime import timedelta, datetime
from bisect import bisect_right
import json
import subprocess as sp
from collections import defaultdict
from tkinter import messagebox

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

def read_cut_times_from_nav(navPath: str):
    p = pl.Path(navPath)
    if (not p.exists()):
        message = navPath + " does not exist. Please provide a valid file path."
        messagebox.showerror("Error", message)
        raise FileExistsError(message)
    elif (not p.is_file()):
        message = navPath + " is not a regular file. Please provide a valid file path."
        messagebox.showerror("Error", message)
        raise FileExistsError(message)
    df = pd.read_table(navPath, usecols=["HEURE", "CODEseq"])
    idx0 = df.index[df["CODEseq"] == "DEBPLO"]
    t0_string = df.at[idx0[0], "HEURE"]
    t0 = datetime.strptime(t0_string, "%H:%M:%S")
    idx1 = df.index[df["CODEseq"] == "FINFIL"]
    t1_string = df.at[idx1[0], "HEURE"]
    t1 = datetime.strptime(t1_string, "%H:%M:%S")
    idx2 = df.index[df["CODEseq"] == "DEBVIR"]
    t2_string = df.at[idx2[0], "HEURE"]
    t2 = datetime.strptime(t2_string, "%H:%M:%S")
    start = t1 - t0
    end = t2 - t0
    return [start, end]

def read_time_offset_from_nav(navPath: str):
    p = pl.Path(navPath)
    if (not p.exists()):
        message = navPath + " does not exist. Please provide a valid file path."
        messagebox.showerror("Error", message)
        raise FileExistsError(message)
    elif (not p.is_file()):
        message = navPath + " is not a regular file. Please provide a valid file path."
        messagebox.showerror("Error", message)
        raise FileExistsError(message)
    df = pd.read_table(navPath, usecols=["HEURE", "CODEseq"])
    idx = df.index[df["CODEseq"] == "FINFIL"]
    t_string = df.at[idx[0], "HEURE"]
    return t_string

def cut_command(input, start, stop, output):
    ffmpeg_cmd = "ffmpeg -loglevel error -i \"{}\" -ss {} -to {} -vcodec h264 -y \"{}\"".format(input, start, stop, output)
    ret = lambda : sp.Popen(ffmpeg_cmd, shell=True).wait()
    try:
        return ret()
    except sp.TimeoutExpired:
        return ret()

def convert_nav_to_csv(
    navPath: str,
    videoName: str,
    outPath: str = None,
    force: bool = False,
    volumeId: int = None,
    email: str = None,
    token: str = None): # -> pd.DataFrame:

    """Converts a Pagure's nav (.txt) file to a csv file readable by Biigle

    Python script that converts a navigation (.txt) file from a Pagure acquisition campain to a metadata (.csv) file readable by Biigle.

    Args:
        navPath (str): Full path to input navigation file (must be a .txt).
        videoName (str): Video filename associated to this nav file. Required for writing in metadata.
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

    df = pd.read_table(navPath, header=0, names = ["file", "", "lat", "lon", "yaw"], usecols=[0, 1, 2, 3, 4], parse_dates={"taken_at" : [0, 1]}, dayfirst=True)
    df.insert(0, "file", videoName, allow_duplicates=True)
    df.to_csv(outFilepath, index=False)

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
    videoPaths: list = None,
    outPath: str = None):

    """Script to convert biigle's video annotations to yolo-formatted images annotations

    Python script that extracts frames annotations from a Biigle's video annotation csv report. Output is a folder containing one csv file for each time of each annotation.

    Args:
        csvPath (str): Full path to Biigle's CSV video annotations file
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
        """
        rowdata = pd.read_csv(csvPath)
        classes = pd.DataFrame(columns=['class_name'])
        classes.index.name = 'class_id'
        annotation_ids = defaultdict(lambda: defaultdict(list))
        annotation_tracks = defaultdict(list)

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
        return annotation_ids, annotation_tracks

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

        frame_count = 0
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
                    with open(pl.Path(out_annotations_path).joinpath(videoname + "_%04d"%frame_count + '.txt'), 'r') as file:
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
                            cv2.imwrite(str(pl.Path(out_images_path).joinpath(path.stem + "_%04d"%frame_count + ".png")), frame)
                frame_count += 1
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

    annotation_ids, annotation_tracks = read_data_from_csv(csvPath)

    # For each video input file
    for videoname in annotation_ids.keys():
        # Keep a counter of keyframes processed to construct output filenames
        kf_count = 0
        # Keep track of annotations being processed in a set of their ids, to interpolate positions in-between key frames
        memdata = set()
        # Sort annotation_ids by keys (i.e. keyframes).
        sorted_ids = sorted(annotation_ids[videoname].items())

        # Browse all keyframes in ascending order
        for k, ids in sorted_ids:
            outdata, memdata = process_keyframe(k, ids, annotation_tracks, memdata)
            outFilepath = pl.Path(outAnnotationsPath).joinpath(videoname + "_%04d"%kf_count + '.txt')
            if outdata.empty:
                print("error: data for timekey {} of file {} is empty".format(k, outFilepath))
                continue
            kf_count += 1
            outdata.to_csv(outFilepath, index=False)

        if videoPaths:
            if (len(videoPaths) == 1):
                p = pl.Path(videoPaths[0])
                if (videoname != p.stem):
                    message = "videopath name {} doesn't match with name read in csv file {}".format(p.stem, videoname)
                    messagebox.showerror("Error", message)
                    continue
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

def interpolate_nav_data(
    nav_days: pd.Series,
    nav_times: pd.Series,
    latitudes: pd.Series,
    longitudes: pd.Series,
    keytime: float,
    timeoffset: pd.Timedelta):
    """Takes info from Pagure nav file (timestamps, latitudes, longitudes) and a keytime in video and interpolates those data at this keytime.

    Python script that takes info from an input navigation file (saved as pandas Series), a keytime (corresponding to time of video where the annotation is closest to the lasers),
    and interpolates the absolute timestamp and GPS position (latitude + longitude) of the annotation (approximated to the pagure's position at this timestamp).

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
    i = bisect_right(nav_times, time)
    if not i:
        raise ValueError
    # get time values wrapping time to compute interpolation coeff and latitude/longitude corresponding to these times
    if (i==0):
        t1, t2 = nav_times[0], nav_times[1]
        lat1, lat2 = latitudes[0], latitudes[1]
        lon1, lon2 = longitudes[0], longitudes[1]
    else:
        t1, t2 = nav_times[i-1], nav_times[i]
        lat1, lat2 = latitudes[i-1], latitudes[i]
        lon1, lon2 = longitudes[i-1], longitudes[i]
    coeff = (time - t1) / (t2 - t1)
    time_delta = t1 + coeff * (t2 - t1)
    time_delta = nav_days[i] + time_delta
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
        # if we already found 2 laser tracks for this video, print error message and return
        if len(laser_tracks[video_filename]) == 2:
            message = "Invalid number of lasers found ({}) for video filename {}. Please verify your data.".format(len(track), video_filename)
            messagebox.showerror(title='Error', message=message)
            return
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
        navPath: str,
        dy_max_str: float,
        laser_tracks: dict,
        laser_label : str = None,
        laser_dist_str: str = None,
        str_timeoffset: str = None,
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
        navPath (str): Full path to input navigation file (.txt or .csv)
        timeOffset (str): time offset to start the timestamps count from nav file (str at format hh:mm:ss)
        outPath (str): Full path to write the output file. Optionnal, by default the input file is overwritten
    """
    import math

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

    data_nav = pd.read_table(navPath)
    nav_days = pd.to_datetime(data_nav["DATE"], dayfirst=True)
    nav_times = pd.to_timedelta(data_nav["HEURE"])
    if "LAT_PAGURE" and "LONG_PAGURE" in data_nav.columns:
        latitudes = data_nav["LAT_PAGURE"]
        longitudes = data_nav["LONG_PAGURE"]
    else:
        latitudes = data_nav["LAT_NAVIRE"]
        longitudes = data_nav["LONG_NAVIRE"]

    if not str_timeoffset:
        time_offset = pd.to_timedelta(nav_times[0])
    else:
        try:
            time_offset = pd.to_timedelta(str_timeoffset)
        except ValueError as e:
            print("error: failed to convert string time {} to timedelta".format(str_timeoffset), e)
            time_offset = pd.to_timedelta(nav_times[0])

    data = pd.read_csv(csvPath)
    out_data = pd.DataFrame(index=data.index,
                            columns=['video_annotation_label_id', 'video_filename', 'label_name', 'label_hierarchy',
                                     'frame', 'timestamp', 'lasers_position_image', 'lasers_distance_image',
                                     'annotation_position_image', 'annotation_GPS_position', 'annotation_size_pixels', 'annotation_size_cm'])

    for row in data.itertuples():
        video_annotation_label_id = row.video_annotation_label_id
        video_filename = row.video_filename
        label_name = row.label_name
        out_data.loc[row.Index, "video_annotation_label_id"] = video_annotation_label_id
        out_data.loc[row.Index, "video_filename"] = video_filename
        out_data.loc[row.Index, "label_name"] = label_name
        out_data.loc[row.Index, "label_hierarchy"] = row.label_hierarchy
        shape_id = row.shape_id
        if shape_id == 7:     # whole frame annotation
            continue
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
        # search for tracks in laser_tracks corresponding to this video
        if not video_filename in laser_tracks:
            message = "No laser annotation for video filename {}. Please verify your data.".format(video_filename)
            messagebox.showerror(title="Error: ", message=message)
            continue
        track_l1 = laser_tracks[video_filename][0]
        track_l2 = laser_tracks[video_filename][1]
        if len(track_l1) == 0 or len(track_l2) == 0:
            message = "At least one laser track is empty for video {}. Please verify your data.".format(video_filename)
            messagebox.showerror(title="Error: ", message=message)
            continue
        dist_min = None
        t_min = None
        l1_min = None
        l2_min = None
        c_min = None
        coords_min = None
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
        # Look in laser tracks for segment-keyframe where annotation should be interpolated with bisect-right method
        kfs_l1 = list(track_l1.keys())
        kfs_l2 = list(track_l2.keys())
        for count, time in enumerate(times):
            coords = points[count]
            coords = coords.split(',')
            try:
                t = float(time)
                # Find the two keyframe pairs tmin and tmax surrounding t in laser keyframes with bisect_right
                pos_l1 = bisect_right(kfs_l1, t)
                pos_l2 = bisect_right(kfs_l2, t)
                if pos_l1 == 0 or pos_l2 == 0 or pos_l1 == len(kfs_l1) or pos_l2 == len(kfs_l2):
                    raise ValueError("pos_l1 = {}, pos_l2 = {} and kfs_l1 = {}, kfs_l2 = {}".format(pos_l1, pos_l2, kfs_l1, kfs_l2))
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
                if dist_min == None or dist < dist_min:
                    dist_min = dist
                    t_min = t
                    c_min = barycenter
                    l1_min, l2_min = icoords_l1, icoords_l2
                    coords_min = coords
            except:     # time == null means there is a gap in annotation, skip and continue on next keyframe
                print("couldn't find interpolation segment keyframes at {} for video {}, annotation {}".format(t, video_filename, video_annotation_label_id))
                continue
        if dist_min == None or t_min == None or l1_min == None or l2_min == None or c_min == None:
            print("Error, couldn't compute one of the output value for annotation {} in video {}".format(video_annotation_label_id, video_filename))
            continue
        if (dist_min > dy_max):
            message = "Found a minimum distance to lasers : {} pixels for annotation n°{} with label {}. It is more than {}% of the video height ({} pixels) which is set as the maximum distance to measure objects." \
                    " Skip this annotation.".format(round(dist_min), video_annotation_label_id, label_name, dy_max_str, dy_max)
            print(message)
            # messagebox.showinfo(title="Info", message=message)
            continue
        timestamp, latitude, longitude = interpolate_nav_data(nav_days, nav_times, latitudes, longitudes, t, time_offset)
        if shape_id != 1:
            # for circles we take diameter (2 * third coordinate) as size
            if shape_id == 4:
                annot_size_px = 2*coords_min[2]
            # for lines we take total length (sum of segment lengths) as size
            elif shape_id == 2:
                annot_size_px = 0
                # zip coordinates as (x, y) pairs then re-zip as points pairs pt1=(x1, y1) and pt2=(x2, y2) to compute segment lengths
                coords_pts = list(zip(coords_min[::2], coords_min[1::2]))
                for pt1, pt2 in zip(coords_pts, coords_pts[1:]):
                    annot_size_px += math.dist(pt1, pt2)
            # for rectangles we take diagonal as size. If it's a polygon with 4 points, assume it's close to parallelogram and take one diagonal
            elif shape_id == 5 or len(coords_min) == 8:
                pt1 = (coords_min[0], coords_min[1])
                pt2 = (coords_min[4], coords_min[5])
                annot_size_px = math.dist(pt1, pt2)
            else:
                messagebox.showerror("Error", "Annotation shape {} not supported, skipped.".format(row.shape_name))
                continue
        laser_dist_px = math.dist(l1_min, l2_min)
        if laser_dist_str and annot_size_px:
            laser_dist = float(laser_dist_str)
            annot_size = annot_size_px * laser_dist/laser_dist_px
        l1_pos = (round(l1_min[0], 2), round(l1_min[1], 2))
        l2_pos = (round(l2_min[0], 2), round(l2_min[1], 2))
        out_data.loc[row.Index, "frame"] = round(t_min, 2)
        out_data.loc[row.Index, "timestamp"] = timestamp
        out_data.loc[row.Index, "lasers_position_image"] = l1_pos, l2_pos
        out_data.loc[row.Index, "lasers_distance_image"] = round(laser_dist_px, 2)
        out_data.loc[row.Index, "annotation_position_image"] = (round(c_min[0], 2), round(c_min[1], 2))
        out_data.loc[row.Index, "annotation_GPS_position"] = latitude, longitude
        out_data.loc[row.Index, "annotation_size_pixels"] = annot_size_px
        out_data.loc[row.Index, "annotation_size_cm"] = annot_size
    out_data.to_csv(outFilepath, index=False)
    return True


def auto_detect_laserpoints(input, output, device):
    from components import detect_laserpoints
    try:
        detect_laserpoints.main([input, 'LP_detection\epoch_7.pth', 'LP_detection\ddod_r50.py', output, 'result.mp4', 'coco.json', '512', '0.5', '0.5', 'mmdet', device, 'True', '4', 'False'])
        return True
    except:
        print("error")
        return False