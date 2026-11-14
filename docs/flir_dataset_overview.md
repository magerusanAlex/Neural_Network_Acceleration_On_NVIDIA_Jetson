
# FLIR ADAS v2 - Dataset Overview

## 1. Scop

Datasetul FLIR ADAS v2 este utilizat pentru evaluarea
detectiei de obiecte in scene de trafic.

Proiectul foloseste initial imaginile RGB.
Imaginile termale sunt pastrate pentru o extensie ulterioara.

## 2. Structura datasetului

| Subset | Numar imagini |
|---|---:|
| images_rgb_train | 10319 |
| images_rgb_val | 1085 |
| images_thermal_train | 10742 |
| images_thermal_val | 1144 |
| video_rgb_test | 3749 |
| video_thermal_test | 3749 |
| Total | 30788 |

## 3. Adnotari

Fiecare subset contine fisierele:

- coco.json
- index.json

Fisierul coco.json este utilizat pentru identificarea
imaginilor, claselor si a obiectelor adnotate.

Distributia completa a adnotarilor pe clase
urmeaza sa fie documentata.

## 4. Teste efectuate

- YOLO11n a fost testat pe bus.jpg.
- Inferenta pe cadre RGB FLIR a functionat.
- A fost generat flir_rgb_detection.mp4.

## 5. Urmatoarele etape

- Validarea PyTorch vs. ONNX pe imagini FLIR.
- Construirea motoarelor TensorRT FP32 si FP16.
- Accelerarea preprocesarii si postprocesarii CUDA.
