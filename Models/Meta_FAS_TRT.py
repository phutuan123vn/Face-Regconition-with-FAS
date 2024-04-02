import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
from itertools import product as product
from PIL import Image
from torchvision import transforms as tf
import copy

# def Transform(img):
#     Trans = tf.Compose([
#         tf.ToTensor(),
#         tf.Resize((256,256)),
#         tf.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])
#     return Trans(img)

Transform = tf.Compose([
        tf.ToTensor(),
        tf.Resize((256,256)),
        tf.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])


class Meta_Pattern(object):
    """
    description: A Retineface class that warps TensorRT ops, preprocess and postprocess ops.
    """
    def __init__(self, engine_path, imgsz=(256, 256)):
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
        # face = np.expand_dims(face, 0)
        trt_outputs = self.infer(image)
        trt_outputs = trt_outputs[0].reshape((1,3,256,256))
        img_color = copy.deepcopy(trt_outputs)
        del trt_outputs
        return img_color
    
    def destroy(self):
        self.cfx.pop()




class HFN(object):
    """
    description: A Retineface class that warps TensorRT ops, preprocess and postprocess ops.
    """
    def __init__(self, engine_path, imgsz=(256, 256)):
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
                
        

    def infer(self, img):

        self.cfx.push()

        self.inputs[0]['host'] = np.ravel(img[0])
        self.inputs[1]['host'] = np.ravel(img[1])
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
    

    def inference(self, image_RGB,image_Tex):
        pixel_map, binary_vector = self.infer((image_RGB,image_Tex))
        cls_score = torch.softmax(torch.from_numpy(binary_vector), 0)[1].item()
        map_score = np.mean(pixel_map)
        score = map_score + cls_score
        score /= 2
        del map_score,cls_score,pixel_map,binary_vector
        return score
    
    def destroy(self):
        self.cfx.pop()


class Meta_FAS:

    def __init__(self) -> None:
        self.meta_pattern = Meta_Pattern('./checkpoint/MetaPattern.trt')
        self.hfn = HFN('./checkpoint/HFN.trt')


    def classify(self,img):
        image_RGB = Image.fromarray(img)
        image_RGB = Transform(image_RGB).numpy()
        image_RGB = np.expand_dims(image_RGB,0)
        img_tex = self.meta_pattern.inference(image_RGB)
        score = self.hfn.inference(image_RGB,img_tex)
        return score,img_tex[0]
    
    def __del__(self):
        self.hfn.destroy()
        self.hfn.destroy()