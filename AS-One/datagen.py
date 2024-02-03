import sys
import argparse
import asone
from asone import ASOne
import torch
import numpy as np
from PIL import Image
import subprocess


def main(args):
    filter_classes = args.filter_classes

    filter_classes = ['person']
    # Check if cuda available
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False
     
    if sys.platform.startswith('darwin'):
        detector = asone.YOLOV7_MLMODEL 
    else:
        detector = asone.YOLOV5M_PYTORCH
    
    detect = ASOne(
        tracker=asone.BYTETRACK,
        detector=detector,
        weights=args.weights,
        use_cuda=args.use_cuda
        )
    # Get tracking function
    track = detect.track_video(args.video_path,
                                output_dir=args.output_dir,
                                conf_thres=args.conf_thres,
                                iou_thres=args.iou_thres,
                                display=args.display,
                                draw_trails=args.draw_trails,
                                filter_classes=filter_classes,
                                class_names=None) # class_names=['License Plate'] for custom weights
    
    # Loop over track_fn to retrieve outputs of each frame 
    for bbox_details, frame_details in track:
        bbox_xyxy, ids, scores, class_ids = bbox_details
        frame, frame_num, fps = frame_details
        """print(bbox_details)
        print(ids)
        print(scores)
        print(class_ids)
        print(frame)
        print(frame_num)
        print(fps)"""
        print(bbox_details)
        for i in range(len(class_ids)):
            if len(bbox_xyxy) > 0:
                if class_ids[0] == 0:
                    f = open('../dataset/labels/val/' + str(frame_num) + '.txt', 'w')
                    x1,y1,x2,y2 = bbox_xyxy[i][0],bbox_xyxy[i][1],bbox_xyxy[i][2],bbox_xyxy[i][3]
                    if x1 < 0:
                        x1 = 0
                    if x2 > 1080:
                        x2 = 1080
                    if y1 < 0:
                        y1 = 0
                    if y2 > 1920:
                        y2 = 1920
                    x,y = (x1+x2)/2,(y1+y2)/2
                    w,h = abs(x1-x2),abs(y1-y2)
                    x,y = x/1080,y/1920
                    w,h = w/1080,h/1920
                    f.write("80 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
                    f.close()
                    break
    
    subprocess.call(["python", "../python/imagen.py"])


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda',
                        help='run on cpu if not provided the program will run on gpu.')
    parser.add_argument('--no_save', default=True, action='store_false',
                        dest='save_result', help='whether or not save results')
    parser.add_argument('--no_display', default=True, action='store_false',
                        dest='display', help='whether or not display results on screen')
    parser.add_argument('--output_dir', default='data/results',  help='Path to output directory')
    parser.add_argument('--draw_trails', action='store_true', default=False,
                        help='if provided object motion trails will be drawn.')
    parser.add_argument('--filter_classes', default=None, help='Filter class name')
    parser.add_argument('-w', '--weights', default=None, help='Path of trained weights')
    parser.add_argument('-ct', '--conf_thres', default=0.25, type=float, help='confidence score threshold')
    parser.add_argument('-it', '--iou_thres', default=0.45, type=float, help='iou score threshold')

    args = parser.parse_args()

    main(args)