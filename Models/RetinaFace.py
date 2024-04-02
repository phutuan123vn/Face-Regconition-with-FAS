import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
from itertools import product as product
from math import ceil
from PIL import Image
import math
from albumentations import Compose, Normalize, Resize

def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

# Transform = tf.Compose([
#         tf.ToTensor(),
#         tf.Resize((256,256)),
#         tf.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
# ])

def pre_process(image):
    pre_process_transform = Compose(
            [                                              #0.229, 0.224, 0.225
                Normalize(mean=(0.485, 0.456, 0.406), std=(1/255, 1/255, 1/255), max_pixel_value=255.0, p=1.0),
            ]
        )
    return pre_process_transform(image=image)["image"]


def alignment_face(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y >= right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = euclidean_distance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle,expand=False))
    return img


class PriorBox(object):
    def __init__(self, image_size=(640, 640)):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def decode(loc, priors, variances=[0.1, 0.2]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


class Retinaface_trt(object):
    """
    description: A Retineface class that warps TensorRT ops, preprocess and postprocess ops.
    """
    def __init__(self, engine_path, imgsz=(640, 480)):
        self.imgsz = imgsz
        self.mean = None
        self.std = None
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

        self.priorbox = PriorBox()   
        priors = self.priorbox.forward()
        self.prior_data = priors.data
        self.var = torch.tensor([0.1, 0.2])
        

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
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def destroy(self):
        self.cfx.pop()

    def inference(self, img_raw, confidence_threshold=0.1, nms_threshold=0.5, top_k=5000, keep_top_k=750):
        # img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # starts = time.perf_counter()
        image, ratio, dwdh = self.letterbox(img_raw, auto=False)
        # dwdh = np.array(dwdh*2)
        ########## TF######################
        image = image.astype(np.float32)
        image -= (104, 117, 123)
        # image = pre_process(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        # print("Time pre-proccess: ",time.perf_counter()-starts)
        ################# #######################
        # starts = time.perf_counter()
        trt_outputs = self.infer(image)
        # print("Time infer: ",time.perf_counter()-starts)
        
        
        # starts = time.perf_counter()
        loc = trt_outputs[0].reshape([1, 16800, 4])
        conf = trt_outputs[2].reshape([1, 16800, 2])
        landms = trt_outputs[1].reshape([1, 16800, 10])
        loc = torch.from_numpy(loc)

        boxes = decode(loc[0], self.prior_data).numpy()
        
        boxes = boxes*640
        boxes = (boxes - np.array(dwdh*2)) / ratio
        
        scores = conf[0][:, 1]
        landms = torch.from_numpy(landms)
        landms = decode_landm(landms[0], self.prior_data, self.var).numpy()

        scale1 = np.array([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                           image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                           image.shape[3], image.shape[2]])

        landms = landms * scale1
        landms = landms.reshape(-1, 5, 2)
        landms = (landms - dwdh) / ratio

        inds = np.where(scores > confidence_threshold)[0]
        crop = None
        bbox = None
        kpt = None
        if len(inds) > 0:
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]
            
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_top_k, :]
            landms = landms[:keep_top_k, :]
            areas = [self.comput_area(box) for box in dets]
            max_index = np.argmax(areas)
            bbox = dets[max_index]
            kpt = landms[max_index]
            bbox = np.clip(bbox, a_min=0, a_max=10000)
            kpt = np.clip(kpt, a_min=0, a_max=10000)
            # x1,y1,x2,y2 coordinate
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            crop = img_raw[y1:y2, x1:x2]
            bbox = [bbox]
            kpt = [kpt]
            # print("Time post-Proccess: ",time.perf_counter()-starts)
            # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        return crop, bbox, kpt

    def comput_area(self,bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        return (x2 - x1)*(y2 - y1)


    def Detect_n_Align(self,img_raw):
        crop,bbox,kpt = self.inference(img_raw)
        if bbox is None: return None,None
        img_align = alignment_face(img_raw,kpt[0][0],kpt[0][1])
        crop1,_, _ =self.inference(img_align)
        if crop1 is not None:
            return crop1,bbox
        else:
            return crop,bbox

    def warmup(self,img):
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        for _ in range(3):
            _,_,_ = self.inference(img)
        return
    
    def Detect_n_Align2(self,img_raw):
        crop,bbox,kpt = self.inference(img_raw)
        if bbox is None: return None,None
        img_align = self.quick_align(img_raw,kpt[0][0],kpt[0][1])
        box = [int(i) for i in bbox[0][:4]]
        face_img = img_align[box[1]:box[3],box[0]:box[2]]
        return face_img,[box]
        

    def quick_align(self,img,left_eye,right_eye):
        left_eye = tuple(int(value) for value in left_eye)
        right_eye = tuple(int(value) for value in right_eye)
        dX = right_eye[0] - left_eye[0]
        dY = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate the center of the eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        # Create an affine transformation matrix for rotation
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
        aligned_face = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        # Image.fromarray(aligned_face)
        return aligned_face
