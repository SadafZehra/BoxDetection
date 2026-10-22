# Real-Time Package Detection on an Industrial Line with YOLOv4-Tiny

Built during my computer vision internship at **The Disrupt Labs** (Karachi).

On a factory line, boxes need to be found in every camera frame so they can be counted, tracked and checked. This project fine-tunes a **YOLOv4-tiny** detector on a custom dataset of factory camera images to detect boxes in real time.

## Key results

Evaluated on the validation set (518 ground-truth boxes), confidence threshold 0.25.

| Metric | Value |
|---|:-:|
| **mAP@0.50** | **95.72%** |
| Precision | **1.00** |
| Recall | 0.85 |
| F1-score | 0.92 |
| Average IoU | 85.19% |
| True positives / false positives / false negatives | 439 / **0** / 79 |
| Inference time | about **15.7 ms per image** |
| Model compute | 6.79 BFLOPs |

**Zero false positives** means the detector never reported a box where there was none, which matters on a production line where false alarms stop the process. The remaining errors are missed boxes (recall 0.85), mostly a matter of lowering the confidence threshold or adding more varied training images.

## Approach

- **Model:** YOLOv4-tiny, a small and fast version of YOLOv4 suited to real-time use, starting from the pretrained `yolov4-tiny.conv.29` weights (transfer learning).
- **Framework:** [Darknet](https://github.com/AlexeyAB/darknet), compiled with CUDA and cuDNN on a Google Colab GPU. The notebook detects the GPU type and sets the right compute architecture for compilation automatically.
- **Data:** custom dataset of factory camera images, labeled and exported in Darknet format through Roboflow. One class: `box`.
- **Configuration:** input 416 × 416, batch 64, subdivisions 24. Following the Darknet guidelines, `max_batches = 2000 × number of classes`, learning-rate steps at 80% and 90% of training, and the filters of each YOLO layer set to `(classes + 5) × 3`.
- **Monitoring:** mAP is computed on the validation set during training, and the best weights are saved automatically.

## How to run

1. Open the notebook in Google Colab and select a GPU runtime.
2. Run the setup cells to clone and compile Darknet with GPU, cuDNN and OpenCV support.
3. Add your own Roboflow API key as an environment variable or Colab secret (do not paste keys into the notebook):
   ```python
   import os
   from roboflow import Roboflow
   rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
   ```
4. Download the dataset in Darknet format, generate the config file, and train:
   ```bash
   ./darknet detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map
   ```
5. Run detection on a new image:
   ```bash
   ./darknet detect cfg/custom-yolov4-tiny-detector.cfg backup/custom-yolov4-tiny-detector_best.weights your_image.jpg -dont_show
   ```

## Main libraries and tools

Darknet · YOLOv4-tiny · CUDA / cuDNN · OpenCV · Roboflow · Google Colab

## Author

**Sadaf Zehra**, MS Data Science, NED University of Engineering and Technology
[GitHub](https://github.com/SadafZehra) · [LinkedIn](https://www.linkedin.com/in/sadaf-zehra-796858230) · [ORCID](https://orcid.org/0009-0009-7530-9006)
