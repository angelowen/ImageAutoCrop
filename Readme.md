# Image Auto-Crop Based on OpenCV & YOLO

## Dependencies
  * opencv
  * numpy
  * opencv-python

## How to use
  * there are two ways to implement,if the first way isn't work as u thought(e.g image/juice.jpg) ,then try the second command.
  1. 
  ```
  wget https://pjreddie.com/media/files/yolov3.weights
  ```
  * `--sq` can be added to crop the square image
  ```
  python yolo_opencv.py --image [img/curry.jpg] --config /
  yolov3.cfg --weights yolov3.weights --classes yolov3.txt [--sq]
  ```
  2. 
  ```
  python auto_crup [img/juice.jpg]
  ```



## Reference
  * https://blog.gtwang.org/programming/python-opencv-auto-crop-and-rotate-scanned-image-tutorial/

  * https://blog.csdn.net/liqiancao/article/details/55670749