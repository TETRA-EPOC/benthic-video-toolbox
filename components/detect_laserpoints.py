import argparse
import math
import time
# import uuid
import json
# import subprocess as sp
from os import remove
import pathlib as pl
try:
    from tqdm import tqdm
    # from tqdm.contrib.concurrent import process_map
    from sahi import AutoDetectionModel
    from sahi import predict
    from sahi.utils.file import save_json, load_json
    # from pycocotools.coco import COCO
    from PIL import Image
    # import concurrent.futures as cf
    import multiprocessing as mp
    import cv2
except ImportError:
    print("importerror")

def process_image(argument):
    input_img, framenumber, args, detection_model, fps = argument

    # opencv frame, convert to PIL image
    img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

    result = predict.get_sliced_prediction(
        img,
        detection_model=detection_model,
        slice_height=args.cropsize,
        slice_width=args.cropsize,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        perform_standard_pred=args.perform_standard_pred,
        verbose=0
    )

    annotation_dict=[]
    for object_prediction in result.object_prediction_list:
        try:
            # convert to coco format
            coco_prediction = object_prediction.to_coco_prediction()
            coco_prediction.image_id = framenumber
            coco_prediction_json = coco_prediction.json
            coco_prediction_json["keytime"] = framenumber/fps if fps else None
            if coco_prediction_json["bbox"]:
                annotation_dict.append(coco_prediction_json)
        except Exception as error:
            print("error processing sahi prediction for frame n° {}: {}".format(framenumber, error))
            continue
    return annotation_dict

def filter_predictions(predictions, width, height, coco, args, framecount, fps, videopath, output_path):
    annotation_id = 0

    if args.save_video:
        # open the video with opencv
        videoIn = cv2.VideoCapture(videopath)
        # create VideoWriter object
        try:
            videoOut = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height), True)
        except:
            print("Error while creating VideoWriter object")
            videoIn.release()
            videoOut.release()
            return

    if args.val:
        videoname = pl.Path(videopath).stem
        val_path = pl.Path(args.output).joinpath('val')
        val_path = str(val_path.joinpath(videoname))

    mem_left = (0, 0)
    mem_right = (0, 0)
    val_count = 0
    for frame_id_str, coco_json in predictions.items():
        detection_pair = ({}, {})
        frame_id = float(frame_id_str)
        try:
            img_dict = {
                "id": frame_id_str,
                # "keytime": float(frame_id)/fps if fps else None,
                "width": width,
                "height": height,
            }
            if not img_dict in coco["images"]:
                coco["images"].append(img_dict)
        except ValueError as e:
            print("error writing image dict:", e)
        coco_json.sort(key=lambda x: float(x["score"]), reverse=True)
        for idx, coco_prediction in enumerate(coco_json):
            # stop if we reached last pred of image (no neighbour left to search)
            if idx == len(coco_json)-1:
                break
            # skip if empty pred or missing bbox
            if not coco_prediction or not 'bbox' in coco_prediction:
                continue
            # calculate center point of bbox: [x+w/2, y+h/2]
            bbox = [float(v) for v in coco_prediction["bbox"]]
            center_pred = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
            # search for another prediction in list on the other half of the image and y-aligned with this one
            for neighbour in coco_json[idx+1:]:
                if not 'bbox' in neighbour:
                    continue
                # comparing both center_y from bboxes to get y-alignment (dy = 0 means detections are aligned) threshold of dy_max = 0.02*video_height
                n_bbox = [float(v) for v in neighbour["bbox"]]
                center_n = (n_bbox[0] + n_bbox[2] / 2.0, n_bbox[1] + n_bbox[3] / 2.0)
                if abs(center_pred[1] - center_n[1]) > 0.02 * height:
                    continue
                # check that each detection is on one side of the image
                if (center_pred[0] - width/2.0) * (center_n[0] - width/2.0) > 0:
                    continue
                if (center_pred[0] - width/2.0) < 0:
                    left = center_pred
                    right = center_n
                else:
                    left = center_n
                    right = center_pred
                # if no previous detection found, keep this one in memory
                if (mem_left == (0, 0) and mem_right == (0, 0)):
                    coco_prediction["id"] = annotation_id
                    neighbour["id"] = annotation_id+1
                    annotation_id+=2
                    detection_pair=(coco_prediction, neighbour)
                    coco["annotations"].extend(detection_pair)
                    mem_left = left
                    mem_right = right
                    break
                # last step to validate both predictions: check that they didn't moved too much compared to previous frame. threshold od dx_max = 0.05*video_width
                elif (math.dist(mem_left, left) < 0.05 * width and math.dist(mem_right, right) < 0.05 * width):
                    coco_prediction["id"] = annotation_id
                    neighbour["id"] = annotation_id+1
                    annotation_id+=2
                    detection_pair=(coco_prediction, neighbour)
                    coco["annotations"].extend(detection_pair)
                    mem_left = ((mem_left[0] + left[0])/2.0, (mem_left[1] + left[1])/2.0)
                    mem_right = ((mem_right[0] + right[0])/2.0, (mem_right[1] + right[1])/2.0)
                    # exit the neighbour'searching loop
                    break
            # exit main loop if we found our lasers pair
            if detection_pair[0] and detection_pair[1]:
                break

        if args.save_video and detection_pair[0] and detection_pair[1]:
            # go to frame_id and read frame from the video
            videoIn.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = videoIn.read()
            # check that there is actually a frame returned
            if not ret:
                print("Error: could not read frame number {} from video".format(frame_id_str))
                videoIn.release()
                videoOut.release()
                return
            # save inference results
            frame = export_visuals(detection_pair, frame)
            videoOut.write(frame)
            if args.val and val_path and val_count < 200:
                if frame_id % max(int(framecount/200), 1) == 0:
                    cv2.imwrite(val_path + '_' + str(detection_pair[0]["keytime"]) + ".png", frame)
                    val_count += 1

    #release the video capture objects
    if args.save_video:
        videoIn.release()
        videoOut.release()

    return coco

def export_visuals(annotations, cv2_img):
    # draw prediction bbox with opencv
    for det in annotations:
        lp = det['bbox']
        pt1 = [int(lp[0]), int(lp[1])]
        pt2 = [int(lp[0]+lp[2]), int(lp[1]+lp[3])]
        try:
            # coco coords format is [x0, y0, w, h] whereas OpenCV uses [x0, y0, x1, y1] (opposite corners)
            cv2.rectangle(cv2_img, pt1, pt2, (0, 0, 255), 2)
            label = det['category_name'] + " " + str(round(det['score'], 4))
            lb_box_width, lb_box_height = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]  # label width, height
            outside = pt1[1] - lb_box_height - 3 >= 0  # label fits outside box
            lb_pt1 = [pt1[0], pt1[1] - 3] if outside else pt1
            lb_pt2 = [pt1[0] + lb_box_width, pt1[1] - lb_box_height - 3 if outside else pt1[1] + lb_box_height]
            cv2.rectangle(cv2_img, lb_pt1, lb_pt2, (0, 0, 255), -1, cv2.LINE_AA) # filled
            cv2.putText(
                cv2_img,
                label,
                (lb_pt1[0], lb_pt1[1]-2 if outside else pt1[1] + lb_box_height + 2),
                0,
                1,
                (255, 255, 255),
                1
            )
        except Exception as error:
            print("drawing laserpoint bbox failed for frame number {}:".format(det['image_id']), error)

    return cv2_img

def get_video_details(file):
    try:
        videoIn = cv2.VideoCapture(file)
        # read in the video dimensions
        height, width = (int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH)))
        # what is the total frame count
        frame_count = int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT))
        framerate = videoIn.get(cv2.CAP_PROP_FPS)
        videoIn.release()
        print("Video frame count = {}".format(frame_count))
        print("Width = {}, Height = {}".format(width, height))
        return width, height, frame_count, framerate
    except:
        print("Failed to read video file {} metadata".format(file))
        return

def get_images_details(paths):
    first = cv2.imread(paths[0])
    print("first img path : {}".format(paths[0]))
    height, width = first.shape[0], first.shape[1]
    nbimages = len(paths)
    print("Nb of images = {}".format(nbimages))
    print("Width = {}, Height = {}".format(width, height))
    return width, height, nbimages

def process_video(argument):
    filepath, output_dir, frame_jump_unit, group_number, args, detection_model, fps = argument

    # open the video with opencv
    videoIn = cv2.VideoCapture(filepath)
    # the sample rate to perform detection, default is 1 frame over 4
    sample_rate = args.pred_sample_rate if args.pred_sample_rate else 4
    # a variable to keep track of the current frame id
    proc_frames = frame_jump_unit * group_number

    if args.parallel:
        # construct temp coco paths
        tmp_path = pl.Path(output_dir).joinpath("coco_{}.json".format(group_number))

    # set initial frame
    videoIn.set(cv2.CAP_PROP_POS_FRAMES, proc_frames)
    success = videoIn.grab()
    if not success:
        print("Process group n° {}  with first frame: {}. Error opening video, exit.".format(group_number, proc_frames))
        videoIn.release()
        return

    # dict of all predictions mapped to corresponding frame number:
    out_dict = {}
    # until there is no frame returned anymore (end of video)
    pbar = tqdm(total=frame_jump_unit, desc="Process group number {}".format(group_number))
    while success and proc_frames < frame_jump_unit * (group_number+1):
        if proc_frames % sample_rate == 0:
            # get frame from video reader
            success, frame = videoIn.retrieve()
            # check that there is actually a frame returned
            if success:
                coco_json = process_image([frame, proc_frames, args, detection_model, fps])
                if not coco_json:
                    print("Error: detection failed for frame n° {}".format(proc_frames))
                    proc_frames += 1
                    success = videoIn.grab()
                    continue
                out_dict[proc_frames] = coco_json
            else:
                print("Error: could not retrieve frame {} from group {}:".format(proc_frames, group_number))
                break
        proc_frames += 1
        success = videoIn.grab()
        pbar.update()
        pbar.refresh()
    pbar.close()

    if args.parallel:
        # save the result as coco json
        save_json(out_dict, str(tmp_path), indent=4)

    #release video resource
    videoIn.release()

    return out_dict

def main(args):
    # print("main ! opencv version: " + cv2.__version__)
    # print(args.input)
    # print("nb of input : {}".format(len(args.input)))

    output_dir = args.output
    video_path = args.input
    print("input video file path : {}".format(video_path))

    width, height, total_frames, framerate = get_video_details(video_path)
    if not (width and height and total_frames):
        print("Error while reading input metadata, please retry.")
        return

    # create output and val folders if not present
    pl.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.val:
        pl.Path(args.output).joinpath('val').mkdir(parents=True, exist_ok=True)

    # build output filenames
    coco_path = str(pl.Path(output_dir).joinpath(args.json_file_name))
    output_path = str(pl.Path(output_dir).joinpath(args.output_file_name))

    print('loading model with device: {}'.format(args.device))
    # load the model
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=args.model_type,
            model_path=args.model_path,
            config_path=args.model_config_path,
            confidence_threshold=args.confidence_threshold,
            device=args.device)
    except ImportError as e:
        print("Error while loading pretrained model:", e)
        return
    print('...model loaded !')

    start_time = time.time()

    coco = {}
    coco["images"] = []
    coco["categories"] = [{"id":int(idx),"name":category} for idx,category in detection_model.category_mapping.items()]
    coco["annotations"] = []
    predictions = {}

    # Parallel the processing of video using python multiprocessing
    if (args.parallel):
        num_processes = int(args.parallel)
        frame_jump_unit =  math.ceil(total_frames/num_processes)
        print("Video processing using {} processes...".format(num_processes))
        print("Float frame jump unit: " + str(frame_jump_unit))

        try:
            with mp.Pool(num_processes) as pool, tqdm(total=num_processes, desc="Inference") as pbar:
                for r in pool.imap(
                    process_video,
                    [[video_path, output_dir, frame_jump_unit, process, args, detection_model, framerate] for process in range(num_processes)]):
                    pbar.update()
                    pbar.refresh()
                pool.close()
                pool.join()
                pbar.close()
        except:
            print("Failed to launch multiprocessing")
            return
        for i in range(num_processes):
            coco_tmp_path = pl.Path(output_path).parent.joinpath("coco_{}.json".format(i))
            if coco_tmp_path.exists():
                coco_json = load_json(coco_tmp_path)
                predictions.update(coco_json)
                remove(coco_tmp_path)
            else:
                print("Error: file not found {}".format(coco_tmp_path))
                return
        if len(predictions) == 0:
            print("No predictions found...")
            return
    else:
        pbar = tqdm(total=total_frames, desc="Inference")
        predictions = process_video([video_path, output_dir, total_frames, 0, args, detection_model, framerate])
        coco_tmp_path = pl.Path(output_path).parent.joinpath("coco_0.json")
        if coco_tmp_path.exists():
            remove(coco_tmp_path)
        pbar.close()

    print("End of detection, go with filtering result")
    coco = filter_predictions(predictions, width, height, coco, args, total_frames, framerate, video_path, output_path)
    if not coco:
        print("Error: failed to write coco annotation file, something went wrong.")
        return
    # save the result as coco json
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=4)

    end_time = time.time()

    total_processing_time = end_time - start_time
    print("Time taken: {} seconds i.e. {} minutes".format(total_processing_time, total_processing_time/60))
    print("FPS : {}".format(total_frames/total_processing_time))

def parse_args():
    parser = argparse.ArgumentParser(description='Performing laser detection on videos')
    # required arguments
    parser.add_argument('--input', '-i', type=str, help='Path to the input video file',required=True)
    parser.add_argument('--model_path', '-w', type=str, help='Path to model weights',required=True)
    parser.add_argument('--model_config_path', '-c', type=str, help='Path to model config file',required=True)
    # optional parameters
    parser.add_argument('--output', '-o', type=str, help='Output directory for the results',default="output")
    parser.add_argument('--output_file_name', type=str, help='File name of the output video file with bbox drew on lasers',default="result.mp4")
    parser.add_argument('--json_file_name', type=str, help='File name of the coco output file (default: coco.json)',default="coco.json")
    parser.add_argument('--cropsize', '-s', type=int, help='The window size that is used for inference', default=512)
    parser.add_argument('--overlap', '-l', type=float, help='Overlap ratio', default=0.2)
    parser.add_argument('--confidence_threshold', '-b', type=float, help='Threshold for the believe value of the model', default=0.1)
    parser.add_argument('--model_type', '-f', type=str, help='Type of framework used', default='mmdet')
    parser.add_argument('--device', help='Device to run inference on (default:cuda)',default="cuda")
    parser.add_argument('--save_video',action='store_true', help='Create output video with bboxes overlaid (default:False). Only frames with detection will be saved so if pred_sample_rate not 1, video will be subsampled.')
    parser.add_argument('--val', action='store_true', help='Validate results with randomly extracted frames from output video, stored in a val folder in output directory (default: False)')
    parser.add_argument('--parallel', '-p', type=int, help='Use parallel processing with the given number of processes. If not provided, sequential processing is used.')
    parser.add_argument('--perform_standard_pred', action='store_true', help='Perform standard prediction on the whole image in addition to sliced prediction (default: False).')
    parser.add_argument('--pred_sample_rate', '-sr', type=int, help='Sample rate used to perform detection on videos with parallel processing (the detection will be performed only every n frames, bboxes are copied for frames with no detection)')
    return parser

if __name__ == '__main__':
    # mp.freeze_support()
    mp.set_start_method('spawn')
    parser = parse_args()
    args = parser.parse_args()
    main(args)