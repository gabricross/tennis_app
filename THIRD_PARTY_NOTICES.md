# Modelos y atribuciones

Tennis Lab implementa una aplicacion local sobre modelos preentrenados de
terceros. No se atribuye su entrenamiento ni distribuye sus pesos en Git.

## Personas: YOLOX-S

- Autores del modelo: equipo de [YOLOX / Megvii](https://github.com/Megvii-BaseDetection/YOLOX).
- Distribucion ONNX: [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox).
- Licencia del modelo distribuido: Apache-2.0.
- El preprocesado RGB y las dimensiones de salida se han contrastado con la
  documentacion de OpenCV Zoo.
- La preparacion conserva una copia de la licencia junto al modelo local.

## Propuesta de pista y pelota opcionales

| Modelo | Distribucion y revision | Procedencia declarada |
| --- | --- | --- |
| ResNet50, 14 puntos de pista | [kjfk/tennis-court-keypoints-mirror](https://huggingface.co/kjfk/tennis-court-keypoints-mirror/tree/68342e926f5d012c79a02f8cc045bb75ca443ec8) | abdullahtarek/tennis_analysis |
| TrackNet, mapa de intensidad | [kjfk/tracknet-tennis-ball-mirror](https://huggingface.co/kjfk/tracknet-tennis-ball-mirror/tree/1cd650376c405fed6ae5103bad704310a6e52dda) | yastrebksv/TrackNet |

Las fichas de los mirrors declaran MIT. Los repositorios originales consultados
no incluyen un archivo LICENSE. Esta declaracion de un mirror no resuelve por
si sola los derechos originales: no se redistribuyen los pesos ni codigo Python
descargado de esos proyectos. Revisa esos derechos antes de una redistribucion
comercial.

La conversion local reconstruye la topologia a partir de los tensores de pesos
verificados y utiliza `weights_only=True`. Los SHA-256, revisions y fichas se
conservan en el directorio privado de modelos. El modo `-SoloJugadores` no
descarga estos dos modelos.

TrackNet se basa en Huang et al., *TrackNet: A Deep Learning Network for Tracking
High-speed and Tiny Objects in Sports Applications*, AVSS 2019,
[arXiv:1907.03698](https://arxiv.org/abs/1907.03698).

## Detector opcional ligero de pelota

- Distribucion: [RJTPP/tennis-ball-detection](https://huggingface.co/RJTPP/tennis-ball-detection),
  revision `83ec86c8e7c42505b4a19ea7f345aa1dfb971b77`, archivo `best.onnx`.
- SHA-256: `0020f0356923712947d223fcbbe053748ee862f5642c963041a6dc67ca8f910d`.
- La ficha del publicador declara Apache-2.0.
- Los metadatos internos del ONNX identifican Ultralytics YOLOv8,
  version 8.2.5, clase `tennis-ball`, y declaran **AGPL-3.0**.
- Estas declaraciones son contradictorias. La ficha de un tercero no
  sustituye la comprobacion de los derechos del modelo base y del
  entrenamiento. Revisa las obligaciones originales antes de una
  distribucion comercial.
- La descarga es opcional, con `setup_models.py --ball-yolo`. No se
  redistribuyen pesos, archivos pickle, codigo Ultralytics ni una
  dependencia de su paquete. La inferencia local usa ONNX Runtime y
  preprocesado/decodificacion propios.

## Multimedia y dependencias

No se incluye video de un partido ni una base de datos personal. Cada persona
debe aportar grabaciones que tenga derecho a analizar o compartir. Los videos
de Commons descargados opcionalmente conservan sus fuentes y licencias fuera
del repositorio.
El benchmark de diez clips conserva SHA-256, autor, URL de origen y
licencia de cada archivo en el almacenamiento privado. Los clips
adicionales de Pexels usan la [licencia Pexels](https://www.pexels.com/license/);
los dos videos Davis Cup son de NaBUru38 (CC BY-SA 4.0) y el de
Blaz Rola de ZooTennis.com (CC BY 3.0). Los derivados se identifican
como recortados, normalizados, sin audio y con overlays anadidos.
La galeria enlaza cada fuente y su licencia; no se publican videos
ni fotogramas en Git.

FFmpeg se obtiene mediante `imageio-ffmpeg`; NumPy, OpenCV, ONNX Runtime, SciPy,
Pillow, FastAPI, Uvicorn y las herramientas opcionales mantienen sus propias
licencias. Este archivo no reemplaza las condiciones de esas dependencias.
