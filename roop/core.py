#!/usr/bin/env python3

import sys
sys.path.append('roop')
import platform
import signal
import sys
import shutil
import glob
import multiprocessing as mp
import os
import torch
from pathlib import Path
from tkinter import filedialog
from opennsfw2 import predict_video_frames, predict_image
from tkinter.filedialog import asksaveasfilename
import webbrowser
import psutil
import cv2
import threading
from PIL import Image, ImageTk
import tqdm

import roop.globals
from roop.swapper import process_video, process_img, process_faces
from roop.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames, rreplace
from roop.analyser import get_face_single

if 'ROCMExecutionProvider' in roop.globals.providers:
    del torch

pool = None
args = {}

signal.signal(signal.SIGINT, lambda signal_number, frame: quit())

sep = "/"
if os.name == "nt":
    sep = "\\"


def limit_resources():
    if args['max_memory']:
        memory = args['max_memory'] * 1024 * 1024 * 1024
        if str(platform.system()).lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check():
    if sys.version_info < (3, 9):
        quit('Python version is not supported - please upgrade to 3.9 or higher')
    if not shutil.which('ffmpeg'):
        quit('ffmpeg is not installed!')
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
    if not os.path.isfile(model_path):
        quit('File "inswapper_128.onnx" does not exist!')
    if '--gpu' in sys.argv:
        NVIDIA_PROVIDERS = ['CUDAExecutionProvider', 'TensorrtExecutionProvider']
        if len(list(set(roop.globals.providers) - set(NVIDIA_PROVIDERS))) == 1:
            CUDA_VERSION = torch.version.cuda
            CUDNN_VERSION = torch.backends.cudnn.version()
            if not torch.cuda.is_available() or not CUDA_VERSION:
                quit("You are using --gpu flag but CUDA isn't available or properly installed on your system.")
            if CUDA_VERSION > '11.8':
                quit(f"CUDA version {CUDA_VERSION} is not supported - please downgrade to 11.8")
            if CUDA_VERSION < '11.4':
                quit(f"CUDA version {CUDA_VERSION} is not supported - please upgrade to 11.8")
            if CUDNN_VERSION < 8220:
                quit(f"CUDNN version {CUDNN_VERSION} is not supported - please upgrade to 8.9.1")
            if CUDNN_VERSION > 8910:
                quit(f"CUDNN version {CUDNN_VERSION} is not supported - please downgrade to 8.9.1")
    else:
        roop.globals.providers = ['CPUExecutionProvider']
    if '--all-faces' in sys.argv or '-a' in sys.argv:
        roop.globals.all_faces = True


def start_processing():
    frame_paths = args["frame_paths"]
    n = len(frame_paths) // (args['cores_count'])
    # single thread
    if args['gpu'] or n < 2:
        process_video(args['source_img'], args["frame_paths"])
        return
    # multithread if total frames to cpu cores ratio is greater than 2
    if n > 2:
        processes = []
        for i in range(0, len(frame_paths), n):
            p = pool.apply_async(process_video, args=(args['source_img'], frame_paths[i:i+n],))
            processes.append(p)
        for p in processes:
            p.get()
        pool.close()
        pool.join()



def start(vid, source_face, params):
    # global pool
    # pool = mp.Pool(args['cores_count'])
    output = []
    for i in tqdm(range(len(vid))):
        frame = vid[i]
        output_frame = process_faces(source_face, frame, all_faces=params.swap_all_faces)
        output.append(output_frame)
    return output
