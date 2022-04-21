'''
Multiple Object Tracking Model Evaluator
Author: Keyon Amirpanahi

Uses py.motmetrics to assess accuracy of bounding boxes detected by a multiple object
tracking model. Input folder is assumed to be in the MOT format:
MOT DATA
-Video 1
--det
---det.txt
--gt
---gt.txt
--seqinfo.ini

Inferences should be in the det.txt file, and ground truth in the gt.txt file.
Data should also be in the MOT format: 
"<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>"
'''
import motmetrics as mm
import pandas as pd
import sys
import os

if len(sys.argv) != 2:
    print("Run as \"python evaluate.py %RELATIVE_PATH_TO_MOT_FORMAT_DATA%\"")
    exit()

if os.path.exists(sys.argv[1]):
    directory = sys.argv[1]
elif os.path.exists(os.path.join(os.getcwd(), sys.argv[1])):
    directory = os.path.join(os.getcwd(), sys.argv[1])
else:
    print("Invalid directory")
    exit()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

for video in os.listdir(directory):
    video_path = os.path.join(directory, video)
    try:
        os.mkdir(os.path.join(video_path, "eval"))
    except:
        pass

    seqinfo = open(os.path.join(video_path, "seqinfo.ini"))
    for line in seqinfo:
        if line.startswith("seqLength"):
            num_frames = int(line.split("=")[1])

    gt_file = open(os.path.join(video_path, "gt/gt.txt"))
    det_file = open(os.path.join(video_path, "det/det.txt"))
    eval_file = open(os.path.join(video_path, "eval/eval.txt"), "w") 

    gt_objects = {f:{"identities": [], "b_boxes":[]} for f in range(1, num_frames + 1)}
    gt_lines = gt_file.readlines()
    for line in gt_lines:
        elements = line.split(",")
        frame_num = int(elements[0])
        id = int(elements[1])
        gt_objects[frame_num]["identities"].append(id)
        gt_objects[frame_num]["b_boxes"].append([float(i) for i in elements[2:6]])

    det_objects = {f:{"hypotheses": [], "b_boxes":[]} for f in range(1, num_frames + 1)}
    det_lines = det_file.readlines()
    for line in det_lines:
        elements = line.split(",")
        frame_num = int(elements[0])
        id = len(det_objects[frame_num]["hypotheses"]) + 1
        det_objects[frame_num]["hypotheses"].append(id)
        det_objects[frame_num]["b_boxes"].append([float(i) for i in elements[2:6]])   

    # Accumulator that will be updated during each frame
    acc = mm.MOTAccumulator()
    for frame_num in range(1, num_frames + 1):
        distances = mm.distances.iou_matrix(gt_objects[frame_num]["b_boxes"], det_objects[frame_num]["b_boxes"])
        acc.update(gt_objects[frame_num]["identities"], det_objects[frame_num]["hypotheses"], distances, frame_num)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=video)
    eval_file.write(str(summary))
    



