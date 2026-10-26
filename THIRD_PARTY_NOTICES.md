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

## Multimedia y dependencias

No se incluye video de un partido ni una base de datos personal. Cada persona
debe aportar grabaciones que tenga derecho a analizar o compartir. Los videos
de Commons descargados opcionalmente conservan sus fuentes y licencias fuera
del repositorio.

FFmpeg se obtiene mediante `imageio-ffmpeg`; NumPy, OpenCV, ONNX Runtime, SciPy,
Pillow, FastAPI, Uvicorn y las herramientas opcionales mantienen sus propias
licencias. Este archivo no reemplaza las condiciones de esas dependencias.
