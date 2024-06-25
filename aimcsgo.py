import sys
import ctypes
import signal

import time

import argparse
import win32con
import win32api

from mss import mss
from pynput import mouse

from models.experimental import attempt_load
from utils.utils import *

import numpy as np

FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = FILE
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)

def calculate_center(xyxy):
    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
    center_x = int((c2[0] - c1[0]) / 2 + c1[0])
    center_y = int((c2[1] - c1[1]) / 2 + c1[1])
    high = int(c2[1]-c1[1])
    return center_x, center_y, high

def getPos(center_list):
    if center_list:
        # 找最近的敌人，识别的框高度越大说明越近
        nearest = None
        old_high = 0
        for pos in center_list:
            x = pos[0]
            y = pos[1]
            high = pos[2]
            if not nearest:
                nearest = (pos[0], pos[1])
            else:
                if high > old_high:
                    nearest = (pos[0], pos[1])
                    old_high = high

        return nearest


@torch.no_grad()
def run(
        weights=ROOT / 'weights/best.pt',  # model path or triton URL
        data=ROOT / 'data/csgo.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
):
    # Load model
    device = torch.device('cpu')
    model = attempt_load(weights=weights, device=device, inplace=True)
    model = model.to(device)
    stride = max(int(model.stride.max()), 32)  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    print(names)

    print('start')

    sct = mss()
    mouse_control = mouse.Controller()

    enemy_body = []
    enemy_head = []
    
    game_window = {'left': 0, 'top': 0, 'width': 1024, 'height': 768}

    while True:
        # copy from class LoadScreenshots
        # mss screen capture: get raw pixels from the screen as np array
        im0 = np.array(sct.grab(game_window))[:, :, :3]  # [:, :, :3] BGRA to BGR

        im = letterbox(im0, new_shape=(640,640))[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(device).float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im.unsqueeze(0)  # expand for batch dim

        # Run model inference
        t1 = time.time()    # begin model inference

        pred = model(im, augment=augment)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)

        t2 = time.time()    # end model inference

        # Process predictions
        det = pred[0]

        s = '%gx%g ' % im.shape[2:]  # print string
        if det is not None and len(det): # 大窗口检测到人
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # 处理检测到的目标
            enemy_body = []
            enemy_head = []
            for *xyxy, conf, cls in det:
                center_x, center_y, high = calculate_center(xyxy)
                if int(cls) == 3:  # t-body
                    enemy_body.append([center_x, center_y, high])
                if int(cls) == 4:  # t-head
                    enemy_head.append([center_x, center_y, high])

            print('%sDetect a frame. (%.3fs)' % (s, t2 - t1))

            firecount = 0
            isfind = 0   # 是否发现敌人的标记
            if (len(enemy_body) != 0) or (len(enemy_head) != 0): # 发现敌人
                isfind = 1

                if len(enemy_head) != 0:
                    pos = getPos(enemy_head)
                else:
                    pos = getPos(enemy_body)
                
                tx = int(pos[0] * 65535 / win32api.GetSystemMetrics(0) )
                ty = int(pos[1] * 65535 / win32api.GetSystemMetrics(1) )
                ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, tx, ty)
                
                print('move')
            
                # print("shoot")
                # win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                # win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

                # time.sleep(0.1)

            #如果发现敌人，调小窗口检测并瞄准开枪，窗口中看不到敌人时退出循环
            while (isfind == 1): 
                smallbox = {'left': 384, 'top': 256, 'width': 256, 'height': 256}

                # copy from class LoadScreenshots
                # mss screen capture: get raw pixels from the screen as np array
                smallimg0 = np.array(sct.grab(smallbox))[:, :, :3]  # [:, :, :3] BGRA to BGR
        
                im = letterbox(smallimg0, new_shape=(256,256))[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous

                im = torch.from_numpy(im).to(device).float()
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im.unsqueeze(0)  # expand for batch dim

                # Run model inference
                t1 = time.time()    # begin model inference

                pred = model(im, augment=augment)
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)

                t2 = time.time()    # end model inference

                print('256x256 Focus! (%.3fs)' % (t2 - t1))

                # Process predictions
                det = pred[0]
            
                enemy_body = []
                enemy_head = []
                if det is not None and len(det):
                    for *xyxy, conf, cls in det:
                        center_x, center_y, high = calculate_center(xyxy)
                        center_x = center_x + smallbox['left']
                        center_y = center_y + smallbox['top']

                        if int(cls) == 3:  # t-body
                            enemy_body.append([center_x, center_y, high])
                        if int(cls) == 4:  # t-head
                            enemy_head.append([center_x, center_y, high])
            
                    if (len(enemy_body) != 0) or (len(enemy_head) != 0):
                        isfind = 1
                        if len(enemy_head) != 0:
                            pos = getPos(enemy_head)
                        else:
                            pos = getPos(enemy_body)

                        tx = int(pos[0] * 65535 / win32api.GetSystemMetrics(0) )
                        ty = int(pos[1] * 65535 / win32api.GetSystemMetrics(1) )
                        ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, tx, ty)
                
                    else: # 没有检测到敌人，退出循环
                        isfind = 0

                    if (isfind==1):
                        print("Shoot focus!")
                        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                        firecount = firecount + 1
                            
                    # print('focus(%sX%s)' % (focuspos[0], focuspos[1]))
                else: # 没有检测到任何人，退出循环
                    isfind = 0  #break 

                # if (firecount > 5):
                #     isfind = 0  #break
            # end 小窗口瞄准循环
                               
            if (firecount>0):
                print ('fire %s' % firecount)

        # else:
        #     time.sleep(0.5)

        enemy_body = []
        enemy_head = []
        # end 大窗口检测循环


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model path or triton URL')
    parser.add_argument('--data', type=str, default=ROOT / 'data/csgo.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    opt = parse_opt()
    main(opt)
