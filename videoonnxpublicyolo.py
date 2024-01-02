import onnxruntime as rt
import onnxruntime
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt
import onnx
import time
#import numpy as np
#from PIL import Image
import io

print(" Onnx Runtime : " + onnxruntime.get_device())

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]



#model_path = "models/ssd_mobilenet_v1_12.onnx"
model_path = "models/tiny-yolov3-11.onnx"

# opening the file in read mode 
my_file = open("models/coco_classes.txt", "r") 

# reading the file 
data = my_file.read() 

# replacing end splitting the text 
# when newline ('\n') is seen. 
data_into_list = data.split("\n") 
print(data_into_list) 
my_file.close() 



import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt

coco_classes = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
}

def draw_detection(draw, d, c):
    """Draw box and label for 1 detection."""
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels.
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = coco_classes[c]
    label_size = draw.textsize(label)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
    draw.text(text_origin, label, fill=color)  # , font=font)



# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

#image = Image.open(img_path)
# input


import cv2
#sess = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
sess = rt.InferenceSession(str(model_path), providers=providers)
# vid = cv2.VideoCapture(0) # For webcam
print(sess.get_inputs()[0].name)
vid = cv2.VideoCapture("rtsp://office:admin1234@192.168.4.54:88/videoMain") # For streaming links
while True:
    rdy,frame = vid.read()
    #print(rdy)
    start = time.process_time()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame, (640, 480))
    # image = Image.fromarray(img)
    image = Image.fromarray(img, 'RGB')
    #img_data = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3)
    #img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    image_data = preprocess(image)
    image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
    #print(image_size)
    #sess = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    #sess = rt.InferenceSession(model_path)
    # we want the outputs in this order'
    # outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]
    output_names = [o.name for o in sess.get_outputs()]
    outputs = output_names
    #print(outputs)
    #result = sess.run(outputs, {"inputs": image_data})
    result = sess.run(outputs, { "input_1" : image_data , "image_shape" : image_size})
    #print(result)
    #print(type(result))
    scores, boxes, indices = result
    # print number of detections
    #print("----------boxes-------------")
    #print(boxes.shape)
    #print(boxes)
    #print("----------scores-------------")
    #print(scores.shape)
    #print(scores)
    #print("----------indices-------------")
    #print(indices.shape)
    #print(indices)
    #print(detection_boxes[0])

    #cv2.imshow('Video Live IP cam',frame)
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])
        
    batch_size = indices.shape[0]
    #draw = ImageDraw.Draw(Image.fromarray(img, 'RGB'))
    #for batch in range(0, batch_size):
    #    for detection in range(0, int(indices[batch])):
    #        c = boxes[batch][detection]
    #        d = scores[batch][detection]
    #        draw_detection(draw, d, c)

    #plt.figure(figsize=(80, 40))
    #plt.axis('off')
    #plt.imshow(img)
    #plt.show()
    #img = cv2.cvtColor(np.array(draw), cv2.COLOR_RGB2BGR)
    cv2.imshow("output", frame)
    print(" Time taken = " + str(time.process_time() - start))
    
    key = cv2.waitKey(1) & 0xFF
    if key ==ord('q'):
        break
 
#    try:
#    except:
#       pass

vid.release()
cv2.destroyAllWindows()

