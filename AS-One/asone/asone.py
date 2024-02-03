import copy
import cv2
from loguru import logger
import os
import time
import threading
import asone.utils as utils
from asone.trackers import Tracker
from asone.detectors import Detector
from asone.recognizers import TextRecognizer
from asone.utils.default_cfg import config
import numpy as np

class ASOne:
    def __init__(self,
                 detector: int = 0,
                 tracker: int = -1,
                 weights: str = None,
                 use_cuda: bool = True,
                 recognizer: int = None,
                 languages: list = ['en'],
                 num_classes=80
                 ) -> None:

        self.use_cuda = use_cuda

        # get detector object
        self.detector = self.get_detector(detector, weights, recognizer, num_classes)
        self.recognizer = self.get_recognizer(recognizer, languages=languages)
    
        if tracker == -1:
            self.tracker = None
            return
            
        self.tracker = self.get_tracker(tracker)

    def get_detector(self, detector: int, weights: str, recognizer, num_classes):
        detector = Detector(detector, weights=weights,
                            use_cuda=self.use_cuda, recognizer=recognizer, num_classes=num_classes).get_detector()
        return detector

    def get_recognizer(self, recognizer: int, languages):
        if recognizer == None:
            return None
        recognizer = TextRecognizer(recognizer,
                            use_cuda=self.use_cuda, languages=languages).get_recognizer()

        return recognizer

    def get_tracker(self, tracker: int):
        tracker = Tracker(tracker, self.detector,
                          use_cuda=self.use_cuda)
        return tracker

    def _update_args(self, kwargs):
        for key, value in kwargs.items():
            if key in config.keys():
                config[key] = value
            else:
                print(f'"{key}" argument not found! valid args: {list(config.keys())}')
                exit()
        return config

    def track_stream(self,
                    stream_url,
                    **kwargs
                    ):

        output_filename = 'result.mp4'
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(stream_url, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details

    def track_video(self,
                    video_path,
                    **kwargs
                    ):            
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details

    def detect_video(self,
                    video_path,
                    **kwargs
                    ):            
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)
        
        # os.makedirs(output_path, exist_ok=True)

        fps = config.pop('fps')
        output_dir = config.pop('output_dir')
        filename = config.pop('filename')
        save_result = config.pop('save_result')
        display = config.pop('display')
        draw_trails = config.pop('draw_trails')
        class_names = config.pop('class_names')

        cap = cv2.VideoCapture(video_path,)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
            logger.info(f"video save path is {save_path}")

            video_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (int(width), int(height)),
            )

        frame_id = 1
        tic = time.time()

        prevTime = 0
        frame_no = 0
        while True:
            start_time = time.time()

            ret, img = cap.read()
            if not ret:
                break
            frame = img.copy()
            
            dets, img_info = self.detector.detect(img, conf_thres=0.25, iou_thres=0.45)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            if dets is not None: 
                bbox_xyxy = dets[:, :4]
                scores = dets[:, 4]
                class_ids = dets[:, 5]
                img = utils.draw_boxes(img, bbox_xyxy, class_ids=class_ids, class_names=class_names)

            cv2.line(img, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(img, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                        225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


            elapsed_time = time.time() - start_time

            logger.info(
                'frame {}/{} ({:.2f} ms)'.format(frame_no, int(frame_count),
                                                 elapsed_time * 1000))
            frame_no+=1
            if display:
                cv2.imshow('Window', img)

            if save_result:
                video_writer.write(img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            yield (bbox_xyxy, scores, class_ids), (im0 if display else frame, frame_no-1, fps)

        tac = time.time()
        print(f'Total Time Taken: {tac - tic:.2f}')
        # kwargs['filename'] = output_filename
        # config = self._update_args(kwargs)
        
        # for (bbox_details, frame_details) in self._start_tracking(video_path, config):
        #     # yeild bbox_details, frame_details to main script
        #     yield bbox_details, frame_details
    
    def detect(self, source, **kwargs)->np.ndarray:
        """ Function to perform detection on an img

        Args:
            source (_type_): if str read the image. if nd.array pass it directly to detect

        Returns:
            _type_: ndarray of detection
        """
        if isinstance(source, str):
            source = cv2.imread(source)
        return self.detector.detect(source, **kwargs)
    
    def detect_text(self, image):
        horizontal_list, _ = self.detector.detect(image)
        if self.recognizer is None:
                raise TypeError("Recognizer can not be None")
            
        return self.recognizer.recognize(image, horizontal_list=horizontal_list,
                            free_list=[])

    def track_webcam(self,
                     cam_id=0,
                     **kwargs):
        output_filename = 'results.mp4'

        kwargs['filename'] = output_filename
        kwargs['fps'] = 29
        config = self._update_args(kwargs)


        for (bbox_details, frame_details) in self._start_tracking(cam_id, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details
        
    def _start_tracking(self,
                        stream_path: str,
                        config: dict) -> tuple:

        if not self.tracker:
            print(f'No tracker is selected. use detect() function perform detcetion or pass a tracker.')
            exit()

        target_fps = config.pop('target_fps')
        fps = config.pop('fps')
        output_dir = config.pop('output_dir')
        filename = config.pop('filename')
        save_result = config.pop('save_result')
        display = config.pop('display')
        draw_trails = config.pop('draw_trails')
        class_names = config.pop('class_names')

        cap = cv2.VideoCapture(stream_path)
        ret,frame = cap.read()
        global result
        result = None
        global k_read
        k_read = True
        if target_fps is not None:
            fps_cooldown = 1/target_fps
        else:
            fps_cooldown = 0

        #create a thread that reads the cam
        def read_cam(lock:threading.Lock,cap):
            #print('thread started')
            global result
            global k_read
            lock.acquire()
            while k_read:
                lock.release()
                ret,frame = cap.read()
                result = ret,frame
                lock.acquire()
            lock.release()

        lock = threading.Lock()

        reading_thread = threading.Thread(target=read_cam, args=(lock,cap))
        reading_thread.start()

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
            logger.info(f"video save path is {save_path}")

            video_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (int(width), int(height)),
            )

        frame_id = 1
        tic = time.time()

        prevTime = 0

        while True:
            time.sleep(fps_cooldown)

            start_time = time.time()

            #print('stop reading')
            lock.acquire()
            k_read = False
            lock.release()

            #print('wait for thread to die')
            reading_thread.join()
            k_read = True

            #print('take the results')
            ret,frame = copy.deepcopy(result) #no need for a lock here, result is accessed only when the thread is dead

            #print('start a new thread')
            reading_thread = threading.Thread(target=read_cam, args=(lock,cap))
            reading_thread.start()

            #ret, frame = cap.read() #original code

            if not ret:
                lock.acquire()
                k_read = False
                lock.release()
                print('stoped')
                break
            im0 = copy.deepcopy(frame)

            bboxes_xyxy, ids, scores, class_ids = self.tracker.detect_and_track(
                frame, config)
            elapsed_time = time.time() - start_time

            logger.info(
                'frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count),
                                                 elapsed_time * 1000))

            if self.recognizer:
                res = self.recognizer.recognize(im0, horizontal_list=bboxes_xyxy,
                            free_list=[])
                im0 = utils.draw_text(im0, res)
            else:
                im0 = utils.draw_boxes(im0,
                                    bboxes_xyxy,
                                    class_ids,
                                    identities=ids,
                                    draw_trails=draw_trails,
                                    class_names=class_names)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                        225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            if display:
                cv2.imshow(' Sample', im0)
            if save_result:
                video_writer.write(im0)

            frame_id += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                lock.acquire()
                k_read = False
                lock.release()
                print('stoped with q')
                break

            # yeild required values in form of (bbox_details, frames_details)
            yield (bboxes_xyxy, ids, scores, class_ids), (im0 if display else frame, frame_id-1, fps)

        tac = time.time()
        print(f'Total Time Taken: {tac - tic:.2f}')


if __name__ == '__main__':
    # asone = ASOne(tracker='norfair')
    asone = ASOne()

    asone.start_tracking('data/sample_videos/video2.mp4',
                         save_result=True, display=False)
