#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2,os
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('-sq', '--square',  action='store_true', default=False,
                help = 'cut the image with square size')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
image = cv2.imread(args.image)
name = os.path.basename(args.image).split('.')
savename = name[0]+'_cut.'+name[1]
os.makedirs('result/' , exist_ok=True)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

maxobj=[]
maxsize = 0
for i in indices:
    
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    if w*h>maxsize:
        maxsize = w*h
        maxobj=box
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

print(maxobj)
# cv2.imshow("object detection", image)
# cv2.waitKey()
    
cv2.imwrite(f"./result/{name[0]}-detection.jpg", image)
cv2.destroyAllWindows()

## Crop image and save main object
x = int(maxobj[0])
y = int(maxobj[1])
w = int(maxobj[2])
h = int(maxobj[3])
image = cv2.imread(args.image)
if args.square:
    l = w if w>h else h
    center_x = x+w/2
    center_y = y+h/2

    if y+l<Height and x+l<Width:
        cropImg = image[y:y+l,x:x+l]
    ## out of bounds solution
    elif y+l>Height and x+l<Width:
        cropImg = image[y:Height,int(center_x-(Height-y)/2):int(center_x+(Height-y)/2)]
    elif y+l<Height and x+l>Width:
        cropImg = image[int(center_y-(Width-x)/2):int(center_y+(Width-x)/2),x:Width]
    else :## two sides out of bounds !Impossible
        pass
        
else:
    cropImg = image[y:y+h,x:x+w]
print(cropImg.shape)
cv2.imwrite(f'./result/{savename}', cropImg)

