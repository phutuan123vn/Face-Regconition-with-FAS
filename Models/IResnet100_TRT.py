import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
from itertools import product as product
from math import ceil
import math
from PIL import Image
from torchvision import transforms
import copy

class Iresnet100(object):
    """
    description: A Retineface class that warps TensorRT ops, preprocess and postprocess ops.
    """
    def __init__(self, engine_path, imgsz=(112, 112)):
        self.imgsz = imgsz
        
        self.cfx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
        # self.base_transform = transforms.Compose([
        #                 transforms.Resize(self.imgsz),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #                 ])
        

    def infer(self, img):

        self.cfx.push()

        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        self.cfx.pop()
        return data
    

    def inference(self, image):
        # image_pil = Image.fromarray(image)
        # face = self.base_transform(image_pil).numpy()
        # face = np.expand_dims(face, 0)
        trt_outputs = self.infer(image)
        ft = copy.deepcopy(trt_outputs[0])
        # ft = np.expand_dims(ft,0)
        return ft