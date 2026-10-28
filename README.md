# Tennis Lab

Aplicacion web **local y funcional** para cargar clips propios de tenis, revisar la pista, ejecutar inferencia, explorar los resultados y descargarlos. Desarrollada para Gabriele Ruggeri.

No es una pagina con un resultado fijo. Cada analisis procesa el archivo cargado y el intervalo seleccionado.

## Instalar desde GitHub en Windows

Requisitos: Windows 10/11, Python **3.14 de 64 bits**, conexion a Internet para
la preparacion inicial y Edge o Chrome. Se recomienda tener al menos 8 GB de
RAM y empezar con clips cortos. La version verificada utiliza CPU.

```powershell
git clone https://github.com/gabricross/tennis_app.git
Set-Location .\tennis_app
.\Preparar_Tennis_Lab.ps1
```

Tambien puedes descargar el ZIP desde GitHub, extraerlo y ejecutar la
preparacion desde esa carpeta. No necesita la demo de futbol ni un video
incluido en el repositorio.

Para una instalacion mas ligera, sin PyTorch ni los modelos de pista/pelota:

```powershell
.\Preparar_Tennis_Lab.ps1 -SoloJugadores
```

Este modo permite marcar la pista manualmente y analizar jugadores. Puedes
ejecutar la preparacion completa mas adelante para anadir la propuesta de
pista y la pelota experimental.

Si Windows bloquea la ejecucion del script, no es necesario desactivar sus
protecciones. Puedes ejecutar manualmente:

```powershell
py -3.14 -m venv "$env:LOCALAPPDATA\TennisLab\.venv"
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" -m pip install -r requirements-runtime-lock.txt
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\setup_models.py --players-only
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\run_server.py --open
```

## Abrir

1. Abre el acceso directo **Abrir Tennis Lab.lnk**.
2. Se inicia una ventana del servidor y se abre `http://127.0.0.1:8768/`.
3. Conserva abierta la ventana del servidor mientras analizas. Ctrl+C o cerrar esa ventana detiene la aplicacion.

Si ya esta en marcha, el acceso directo abre la instancia existente sin reiniciarla.

Alternativa desde PowerShell, dentro de esta carpeta:

```powershell
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\run_server.py --open
```

**No necesita una cuenta ni servicios de inferencia de pago.** El codigo
puede descargarse desde GitHub; el servidor de analisis sigue siendo
privado y local. Consume recursos y electricidad de tu ordenador.
La direccion localhost solo funciona en este ordenador. GitHub Pages no
ejecuta este motor Python ni procesa clips.

## Primer analisis

1. Carga un MP4, MOV, WebM, MKV o AVI de hasta 300 MB. El original debe durar entre 1 segundo y 30 minutos.
2. Elige **Pista + jugadores + pelota** para ver las tres capas juntas en un unico analisis (seleccion inicial si hay motor de pelota). **Solo pista y jugadores** excluye explicitamente la pelota.
3. Selecciona el inicio y entre **1 y 120 segundos** de analisis; la interfaz propone hasta 30 segundos. El resto del video no se analiza. No se trunca sin indicartelo.
4. Actualiza el fotograma si has cambiado el inicio.
5. Usa **Proponer pista con IA** o marca las cuatro esquinas exteriores de dobles, en este orden:
   - Arriba izquierda.
   - Arriba derecha.
   - Abajo derecha.
   - Abajo izquierda.
6. Arrastra los puntos para corregirlos o utiliza sus coordenadas en porcentaje. El esquema numerado indica que referencias estas marcando.
7. Revisa **todas las lineas turquesas superpuestas**, no solo el rectangulo verde. La interfaz impide confirmar si el apoyo orientativo en lineas blancas es menor de 0,3 o la proyeccion es invalida. Ese apoyo es una comprobacion de coherencia, no precision ni garantia de calibracion correcta. La propuesta del modelo nunca inicia el analisis sin tu revision.
8. Explora el resultado o vuelve mas tarde a la biblioteca.

### Si no se ven las esquinas exteriores

En **Referencias visibles para calibrar**, prueba primero **Rectangulo de
saque completo: ambos lados** si se ven las dos lineas de saque y los dos
laterales de individuales: marca sus cuatro intersecciones exteriores.
Es una referencia mas amplia que una sola columna, sin relajar el rechazo
de rectangulos demasiado pequenos. Los fondos siguen siendo extrapolados.

Tambien puedes seleccionar la columna
izquierda o derecha de cuadros de saque. Marca las cuatro intersecciones
de las dos lineas de saque con la linea central y el lateral de individuales
de esa columna, en orden horario desde el punto lejano izquierdo.

El rectangulo abarca ambos lados de la red. **La cinta superior de la red
no es una referencia del suelo y no debe utilizarse.** Si las cuatro
intersecciones tampoco se ven o estan tapadas, elige otro inicio del clip.

Este modo extrapola la geometria fuera del rectangulo visible. Se identifica
como calibracion parcial en la configuracion y en las advertencias del
resultado. No convierte esas posiciones en mediciones fisicas validadas.
Cambiar la referencia borra los puntos anteriores y exige confirmarlos
de nuevo. Las calibraciones existentes conservan el comportamiento anterior.

El rectangulo verde de saque **no es el contorno completo de la pista**.
Ahora se muestra un esquema con sus cuatro puntos, y todas las lineas
proyectadas se dibujan en turquesa discontinua para advertir de la extrapolacion.
Una referencia parcial que proyecte parte del campo a traves del horizonte
se rechaza: no se dibuja una pista invertida como si fuera valida.
El apoyo sobre lineas exige tambien contraste local de trazo: el suelo
claro de una pista no debe contarse entero como pintura blanca. Aun asi,
la red, las vallas y otros trazos pueden producir coincidencias falsas;
el valor no sustituye la revision visual de todas las lineas.

En calibracion parcial, la camara se comprueba fotograma a fotograma con
flujo optico bidireccional y homografias robustas. Los jugadores se excluyen
de las referencias estaticas desde el primer fotograma. Si se pierde el
registro, se buscan correspondencias ORB con la toma inicial y se exige
ademas apoyo en las lineas de la pista antes de volver a mostrar el mapa.
Los fotogramas sin ajuste fiable siguen siendo huecos, no trayectorias inventadas.
Estas comprobaciones proyectivas, el flujo inverso y la recuperacion verificada
tambien se aplican ahora a la calibracion exterior completa.

### Motores de pelota

El modo **Pista + jugadores + pelota** permite elegir explicitamente un motor. Las
configuraciones anteriores conservan TrackNet; no se sustituye un modelo
ausente por otro silenciosamente.

- **TrackNet temporal**: una inferencia cada tres fotogramas consecutivos.
  Entre inferencias solo se aceptan movimientos de flujo optico
  bidireccional que superen los controles de error y continuidad. No se
  mantiene una marca fija sobre fotogramas sin evidencia nueva.
- **YOLOv8n tenis, experimental ligero**: detector especifico de pelota,
  con inferencia en **cada fotograma**, preservacion de aspecto mediante
  letterbox RGB, NMS y asociacion temporal. El movimiento de componentes
  amarillos, compensando el desplazamiento de camara, ayuda a priorizar
  candidatos frente a pelotas de reserva; no garantiza identificar la
  pelota activa.
- **YOLO multiescala**: usa los mismos pesos a 640 y 960 en cada fotograma.
  Los centros duplicados se fusionan antes de la asociacion temporal:
  dos escalas no cuentan como dos observaciones independientes de movimiento.
  Es la seleccion inicial de la web cuando estan instalados los pesos YOLO,
  y consume mas CPU. Puedes escoger el ligero para reducir el coste.
  La API conserva sus valores por defecto anteriores; los trabajos guardados
  con `yolo` siguen utilizando una sola escala. El nuevo valor explicito
  de `ball_backend` es `yolo_multiscale`.

El segundo motor se instala aparte, sin descargar codigo Python ni
instalar Ultralytics:

```powershell
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\setup_models.py --ball-yolo
```

Los pesos ONNX se verifican por SHA-256. **La ficha declara Apache-2.0 y
los metadatos embebidos del modelo declaran AGPL-3.0**. Esta discrepancia
se conserva en su ficha local: revisa los derechos originales antes de
redistribuirlo comercialmente. No se incluyen pesos en Git.

Todos los modos de pelota exigen una identidad con al menos tres
observaciones y movimiento coherente, compensando la camara. Un candidato
estatico fuerte no basta. Se rechazan desplazamientos compatibles con
ruido estacionario; perder el registro obliga a confirmar otra vez.
Las posiciones con tres candidatos casi inmoviles durante al menos
0,06 s se recuerdan durante hasta 5 s sin nueva evidencia, compensando la
camara. Las coincidencias a 3 pixeles del modelo se excluyen antes de
asignar identidades: una pelota de reserva no debe heredar la historia
de movimiento de la activa cuando caduca su identidad anterior.
La evidencia estatica se acumula por posicion, independientemente de
la asignacion de la pelota activa. Se borra si se pierde el registro
de camara. Una pelota sostenida o casi quieta puede quedar sin marcar;
si sale de esa vecindad debe volver a superar los controles de movimiento.
Las estelas usan identidad y continuidad explicitas: no unen huecos,
saltos rechazados ni muestras con identidades distintas. Una identidad
estimada erroneamente sigue siendo posible. Se prefieren huecos a marcar una
pelota de reserva. Una pelota sostenida o casi inmovil tambien puede
quedar sin marcar; no se garantiza la identidad en todos los encuadres.
Al reprocesar las detecciones neuronales cacheadas de las diez fuentes
anteriores, la revision limitada paso de 13 posiciones correctas y 2
incorrectas a 11 correctas, 0 incorrectas y 4 huecos sobre esas mismas
15 etiquetas visibles. Es un intercambio deliberado entre cobertura y
rechazo de candidatos falsos, no un test ciego ni una precision general.

### Evaluacion local con grabaciones de iPhone

Se han preparado dos grabaciones desde el fondo de **45 y 90 segundos** de
[CalTennis](https://huggingface.co/datasets/demalenk/caltennis), revision
`4af17dd4d398aad4116a35b5a74de4388658d356`. El
[articulo del dataset](https://arxiv.org/abs/2606.20542) describe camaras
principales de iPhone 14 o posteriores, HD a 60 Hz y tripodes de alrededor
de 1,65 m. No son tomas de tribuna elevada. Los recortes locales se
normalizaron a 1280 x 726 y 30 fps. Son dos grabaciones de la misma
sesion/pista, no dos partidos independientes.

Credito: Ilona Demler, Xinran Xie, Blake Werner, Anna Szczuka y Pietro
Perona; CalTennis (2026). **La licencia de los videos es CC BY-NC 4.0**,
no la licencia del articulo. Se utilizan solo para evaluacion local no
comercial, con atribucion en metadatos y resultados. No se incluyen
videos, fotogramas, etiquetas derivadas ni pesos entrenados con ellos en Git.
Una eventual demo comercial requiere material con derechos adecuados.

La calibracion se reviso sobre referencias fisicas del suelo; no se
acepto automaticamente la geometria de origen. Se conserva la advertencia
de extrapolacion parcial. Los numeros de apoyo en pintura no certifican
distancias ni velocidades.

La revision manual de 24 fotogramas produjo **8 posiciones claramente
visibles de pelota activa**. Los ambiguos no se cuentan como ausencia.
Una etiqueta inicial adicional se retiro cuando siete fotogramas vecinos
demostraron que era una pelota de reserva inmovil; esa adjudicacion queda
documentada. No es un conjunto de test ciego ni representativo.

En ventanas de hasta 20 fotogramas previos a cada etiqueta, el seguimiento
ligero localizo 4/8 posiciones; a 960 o 1280, 5/8; y la fusion 640+960,
7/8, sin salidas incorrectas en esas etiquetas, antes del refuerzo de
memoria estatica. El hueco restante fue el
primer fotograma, que no permite confirmar movimiento. **Son resultados
de diagnostico en ventanas, no la precision del video completo.**
La alternativa de flujo optico en huecos YOLO no mejoro la muestra y no se
activo. El nuevo modo multiescala se contrasto tambien sobre los diez
fragmentos anteriores completos de 4 s: antes de reforzar el rechazo de
reservas obtuvo 12/15 posiciones correctas, 0 incorrectas y 3 huecos.
La version final con memoria estatica conserva **11/15 correctas,
0 incorrectas y 4 huecos** al reprocesar esas mismas medidas neuronales.
Es un filtro mas conservador, no una mejora uniforme de cobertura.

La revision del primer recorrido largo encontro 18 marcas sobre una
reserva fija que no aparecian en la pequena muestra de etiquetas.
Por eso no se utiliza el 7/8 como prueba de fiabilidad general.
Se reprodujo la transferencia incorrecta de identidad, se corrigio
el productor y se comprobaron de nuevo las secuencias problematicas.

#### Resultados de los dos recorridos completos corregidos

Se ejecutaron los dos intervalos completos con **YOLO multiescala y la
memoria estatica final**, a traves de la cola y del exportador reales:

| Grabacion | Duracion normalizada | Fotogramas | Cobertura pelota | Jugador arriba | Jugador abajo |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 44,967 s | 1349 | 16,5 % | 71,6 % | 100 % |
| B | 89,967 s | 2699 | 45,2 % | 98,9 % | 96,2 % |

La normalizacion produjo un fotograma menos por recorte. Se verifico
la igualdad de indices y duraciones entre original normalizado, MP4
anotado, JSON y muestras; se decodificaron los **4048 fotogramas** anotados.
Tambien se contrastaron todas las filas de pelota y las 2532 observaciones
CSV de jugadores con sus JSON, las identidades y la continuidad.
Hubo registro de camara valido en todas las muestras de ambos clips;
eso no certifica que la calibracion sea metricamente exacta.

Sobre las mismas ocho posiciones manuales visibles, esta vez tomadas del
**recorrido completo**, hubo **6 localizadas, 0 posiciones incorrectas y
2 huecos**: el primer fotograma de B y su fotograma 105. No es el 7/8 de
las ventanas cortas anteriores. El filtro conservador pierde una de esas
observaciones; esta pequena muestra revisada no mide la precision general
ni permite afirmar que todos los demas candidatos sean correctos.

En A, la auditoria de todas las salidas paso de **18 marcas sobre la
reserva fija a ninguna marca cercana a esa reserva**. En B se revisaron
los ocho puntos cercanos a la reserva conocida, alrededor de los
fotogramas 2310-2314 y 2455-2457: las imagenes originales vecinas muestran
pasos de la pelota en movimiento, no una toma de la reserva. La reserva
confirmada en el fotograma 2650 tampoco se selecciono. Es una auditoria
de casos concretos, no una garantia de ausencia de otros errores.

Para el jugador cercano de B se estimaron **47,8 m observados durante
85,0 s evaluables**, media de 2,0 km/h y pico de ventana de 10,0 km/h.
El jugador lejano tuvo deteccion frecuente, pero solo **31,5 s evaluables
para movimiento**: la extrapolacion y la resolucion descartan muchas
posiciones. No se debe confundir cobertura de cajas con disponibilidad
o precision de metros y km/h.

En CPU AMD Ryzen 5 PRO 7530U, el recorrido B necesito **1598,66 s
(26 min 39 s), incluyendo render**, unas 17,8 veces su duracion.
El pico de RSS muestreado cada segundo del arbol de procesos fue
**782,9 MiB** en B y **791,0 MiB** en A; no es memoria movil ni un maximo
continuo garantizado. El tiempo transcurrido registrado para A fue
26054,01 s durante la continuacion entre sesiones. No se acepta como
medicion representativa de rendimiento: los forwards neuronales sumaron
599,36 s y el tiempo de CPU muestreado fue 2289,95 s; no se atribuye toda
esa diferencia al modelo ni se inventa una duracion corregida.

La revision final paso **85 pruebas unitarias, 49 comprobaciones API y
36 de navegador real**, ademas de sintaxis y compatibilidad de dependencias.
Son controles de regresion y funcionamiento, no pruebas de precision
del detector. Los informes, fotogramas y caches de esta evaluacion
permanecen fuera del repositorio por la licencia de las grabaciones.

No se ha hecho fine-tuning ni se han cambiado pesos: una muestra tan
pequena no permite entrenar y validar de forma fiable. Para una fase de
entrenamiento hacen falta anotaciones suficientes, negativos de pelotas
de reserva y separacion por grabaciones/sesiones con derechos adecuados.
No se promete deteccion perfecta, especialmente con oclusiones o pelotas
de pocos pixeles.

### Metros y km/h de jugadores (JSON version 3)

La pista usa 23,77 x 10,97 m. **Las lineas de saque estan a 6,40 m de la
red, no del fondo**: sus coordenadas longitudinales son 5,485 y 18,285 m.
Se corrigio la geometria anterior de las referencias parciales. Los
analisis anteriores no se convierten en medidas fisicas: hay que
reanalizarlos. El mapa web conserva su geometria antigua al abrir JSON
v1/v2; los nuevos resultados contienen `court_geometry`.
Referencia: [ITF, reglas del tenis, regla 1](https://www.itftennis.com/en/about-us/governance/rules-and-regulations/).

Los pies estimados por las cajas se proyectan al suelo usando la camara
de cada instante. `kinematics.py` calcula distancia observada, tiempo
evaluable, velocidad media y pico estimado de una ventana local. Usa
mediana de tres observaciones, anclas de desplazamiento y una zona
muerta de al menos 0,20 m, ampliada segun la proyeccion de 2 pixeles.
La velocidad local es el incremento de esa misma distancia observada
en una ventana continua de 0,3-0,6 s, no una derivada ruidosa entre dos
cajas consecutivas. Un ancla sin movimiento resoluble se renueva tras
0,6 s. El pico es el mayor promedio de ventana, no una velocidad instantanea.
Se excluyen calibraciones con apoyo insuficiente, resolucion peor que
0,75 m, saltos incompatibles con 12 m/s y huecos mayores de 0,25 s.
Ni una perdida de camara entre muestras ni una oclusion crean recorrido.
Los desplazamientos por debajo de la resolucion no producen un km/h
inventado. La parte inicial de cada segmento requiere observaciones
suficientes; puede haber distancia disponible sin velocidad local.

Son **estimaciones**, no medidas certificadas: las cajas no son sensores
de pies, la camara puede tener distorsion y una calibracion parcial
extrapola fuera de sus referencias. La suma no es la distancia completa
del partido cuando faltan observaciones. Una prueba analitica de 5 m/s
recupera 18 km/h; eso valida las formulas, no la precision en video real.
El MP4, la interfaz, la comparacion y `observations.csv` muestran las
metricas y sus huecos. `motion_status` explica filtros o calentamiento.

Por decision explicita del usuario, **no se calcula velocidad de pelota**.
`summary.ball_speed` devuelve `available: false`, `speed_kmh: null` y
la explicacion. La homografia del suelo no mide su vuelo; se necesita
calibracion 3D o mediciones adicionales. El CSV de pelota deja esa
columna vacia y conserva `track_id` y `continuous` para auditar la estela.

### Que clip funciona mejor

- Individuales, con la pista completa o las cuatro referencias interiores
  elegidas realmente visibles y la camara situada hacia el fondo.
- Una sola toma estable, sin repeticiones ni cambios de plano.
- Preferiblemente 720p o 1080p, sin graficos encima de los jugadores.
- Un fragmento de 5 a 15 segundos para la primera prueba.

No se admite cualquier angulo como si fuese equivalente. Los primeros planos, dobles, tomas laterales recortadas o jugadores tapados pueden dar cobertura baja o impedir el analisis.

El boton opcional **Probar con mi demo de tenis anterior** solo aparece si
existe `Demo_Tennis_Tracker.mp4` en la carpeta superior o si configuras
`TENNISLAB_EXAMPLE_VIDEO` con la ruta de un ejemplo propio. No se incluye ese
archivo en GitHub. Si el ejemplo ya tiene overlays, sirve para comprobar
el flujo, no para medir precision; los contadores originales no los calcula
esta aplicacion.

## Resultados disponibles

- MP4 anotado, sin audio.
- Comparacion con el fragmento original normalizado, conservando el instante del video.
- Enlace **Comparar ambos** para reproducir original y resultado juntos en una
  pagina independiente, con las advertencias y la atribucion de la fuente.
- Minimapa de apoyos estimados, con margenes fuera de las lineas.
- Densidad de observaciones y filtro por mitad de pista.
- Cobertura temporal de jugador de arriba, jugador de abajo y pelota.
- Distribucion de observaciones entre fondo, media pista y red.
- CSV de jugadores; CSV adicional de pelota en modo completo.
- JSON con configuracion, muestras, matrices, diagnosticos y advertencias.
- ZIP con los archivos del analisis.
- Biblioteca persistente, cola de trabajos y cancelacion.
- Borrado de un clip y todos sus resultados desde la biblioteca, con confirmacion.

Las etiquetas **arriba/abajo son mitades de la imagen**, no reconocimiento de identidad. Un cambio de lado no se interpreta como continuidad del mismo deportista.

### Interpretacion

**Cobertura no es precision.** Es el numero de muestras con observaciones dividido entre las muestras analizadas. Las zonas se calculan solo a partir de esas observaciones.

No se rellenan jugadores ocultos para mejorar el porcentaje. El render interpola entre observaciones contiguas; no reconstruye posiciones durante ausencias largas.

La densidad muestra el numero de observaciones del clip completo. No representa esfuerzo, posesion ni un diagnostico.

El modo pelota devuelve candidatos observados en pixeles de imagen. La intensidad de TrackNet no es una probabilidad calibrada. La pelota **no se proyecta al suelo**, porque normalmente esta en el aire.
El score de YOLO tampoco es una probabilidad calibrada de acierto.
Desde la version 2, el JSON contiene `ball_samples`, una entrada por fotograma,
incluidos los huecos. La cobertura de pelota utiliza ese denominador,
no el muestreo de jugadores. `ball.csv` conserva metodo, intensidad
TrackNet cuando corresponde y score del detector YOLO por separado.
Los resultados antiguos siguen siendo legibles.

El minimapa del MP4 incluye los margenes detras de las lineas de fondo
y a los lados, igual que la seleccion de jugadores y el mapa web.
No se ocultan ni se fuerzan dentro de las lineas posiciones de juego
legitimas fuera del rectangulo reglamentario.

**No se calculan** velocidades fiables, distancias fisicas validadas, marcador, botes, numero de golpes, ganador del punto, lesiones ni decisiones dentro/fuera.

Si se pierde el registro de camara, el minimapa deja de mostrar esas
posiciones. El modo parcial puede recuperar el mismo encuadre mediante
referencias estaticas verificadas, pero no resuelve cambios de plano
arbitrarios. Si no recupera una toma fiable, recorta otra y calibra de nuevo.

## Tiempo de procesamiento

Esta version utiliza CPU y no funciona en tiempo real.
El coste y la latencia dependen especialmente del motor de pelota elegido.

En la comprobacion final con YOLO multiescala, un fragmento de dos
segundos necesito 25,4 segundos incluyendo render. El recorrido completo
de 89,967 segundos necesito 26 min 39 s en este ordenador; los detalles
y limites estan en la evaluacion de iPhone anterior. No extrapoles el
resultado de un intervalo corto a cualquier partido: influyen el
contenido, las recuperaciones del jugador lejano y los recursos libres.

**No se ha ejecutado la inferencia en un telefono ni hay una app nativa.**
La interfaz se ha comprobado a 390 px de ancho, pero eso no demuestra
rendimiento movil. No se pueden deducir tiempo, bateria, temperatura o
memoria de un telefono a partir de estas mediciones CPU de Windows.
No se han usado GPU de nube ni servicios de inferencia de pago; el
consumo energetico local no se ha medido.

Empieza con intervalos cortos. Cargar un video largo no implica que todo vaya a procesarse.

Solo hay un analisis ejecutandose y hasta tres trabajos adicionales en cola. La cancelacion de un trabajo en ejecucion puede esperar a que termine la inferencia actual.

## Privacidad y almacenamiento

Por defecto, todos los archivos de usuario y modelos estan en:

```text
%LOCALAPPDATA%\TennisLab\
  .venv\                  Entorno Python aislado
  models\                 Pesos locales y fichas de procedencia
  data\library.sqlite3    Biblioteca y estados persistentes
  data\clips\<uuid>\      Original, vistas previas y resultados de cada clip
  verification\           Informes y capturas de las pruebas
```

Esta ubicacion esta fuera de la carpeta de Escritorio sincronizada por OneDrive. La aplicacion no envia tus clips a terceros ni utiliza recursos web de terceros durante el uso. La descarga inicial de paquetes y modelos si requiere Internet.

El codigo de la aplicacion puede estar en tu Escritorio; eso no cambia la ubicacion privada de los videos. Si exportas resultados a una carpeta sincronizada, tu cliente de sincronizacion puede subirlos: esa ubicacion la eliges tu.

El servidor escucha solo en `127.0.0.1`. Las operaciones de escritura validan la sesion y el origen. No esta preparado ni configurado para exponerse directamente a Internet.

Si se interrumpe el servidor, los trabajos pendientes o en curso se marcan como interrumpidos en el siguiente inicio. Los resultados terminados se conservan. Los trabajos fallidos o cancelados no se presentan como analisis correctos vacios.

## Arquitectura

- FastAPI + Uvicorn: API local y entrega de la interfaz.
- JavaScript, HTML y CSS locales: sin Node, compilacion ni servicios de pago.
- SQLite: biblioteca, configuraciones, estados y resumenes.
- FFmpeg: validacion, fotogramas, normalizacion y H.264.
- YOLOX-S: deteccion de personas.
- ResNet50: propuesta de cuatro esquinas a partir de 14 puntos de pista, seguida de ajuste sobre lineas blancas.
- Flujo optico + RANSAC: controles de movimiento de camara.
- Seleccion por mitad, posicion y continuidad: jugadores.
- TrackNet: candidatos temporales de pelota.
- ONNX Runtime / OpenCV: inferencia CPU; PyTorch solo se necesita para preparar las conversiones.

El detector esta incluido en `detection.py`: el repositorio es autonomo
y no importa archivos de una carpeta hermana ni de la demo de futbol.

## Preparacion y reproduccion

Para instalar, restaurar modelos o generar de nuevo el acceso directo:

```powershell
.\Preparar_Tennis_Lab.ps1
```

No necesita permisos de administrador. No evita ni cambia las politicas de ejecucion de tu sistema.

La preparacion completa utiliza las versiones exactas de
`requirements-lock.txt`. La ligera fija las dependencias directas de
ejecucion en `requirements-runtime-lock.txt`. Se verifican los SHA-256 de
los pesos; la conversion de pista y pelota se realiza en procesos separados
para limitar la memoria.

Opciones de entorno:

| Variable | Uso |
| --- | --- |
| `TENNISLAB_HOME` | Directorio privado de modelos, entorno y datos. Por defecto, AppData local. |
| `TENNISLAB_PORT` | Puerto del servidor local; por defecto, `8768`. |
| `TENNISLAB_EXAMPLE_VIDEO` | Ruta a un ejemplo propio opcional; no se descarga ni publica. |

Por ejemplo, para un entorno independiente:

```powershell
$env:TENNISLAB_HOME = Join-Path $env:LOCALAPPDATA 'TennisLabPruebas'
$env:TENNISLAB_PORT = '8770'
.\Preparar_Tennis_Lab.ps1 -SoloJugadores
```

Para desarrollo:

```powershell
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" -m unittest discover -s tests -v
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" -m pip check
```

Con el servidor ya iniciado:

```powershell
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" tools\verify_api.py --video "C:\MisVideos\tenis.mp4" --corners 0.27 0.18 0.73 0.18 0.90 0.79 0.10 0.79
```

Las coordenadas del ejemplo deben sustituirse por las de tu grabacion
(normalizadas entre 0 y 1). El clip debe contener al menos tres segundos de
individuales visibles; `--start` selecciona su inicio y `--url` otro puerto
local. Los scripts no necesitan el video original del desarrollador.
Para una referencia interior, anade
`--calibration-reference service_boxes`, `service_left` o `service_right` y utiliza sus
cuatro coordenadas visibles, no las de las esquinas exteriores.

La prueba de navegador necesita Playwright y Edge. Instala
`requirements-dev.txt` y ejecuta `tools\verify_browser.py` con las mismas
opciones `--video`, `--corners`, `--start` y `--url`. Las pruebas crean clips
identificados como pruebas; la prueba de interfaz elimina el suyo al acabar.
La prueba API conserva su resultado positivo para inspeccion en la biblioteca.
Usa `--players-only` en ambas herramientas para comprobar una instalacion
ligera sin los modelos opcionales.
Para verificar la pelota ligera, anade `--ball-backend yolo`; para la
fusion de escalas, `--ball-backend yolo_multiscale`. La prueba
API comprueba el motor solicitado, una muestra de pelota por fotograma
y los campos de su CSV; la prueba de navegador selecciona y ejecuta
realmente el modo pelota cuando no se usa `--players-only`.

### Pruebas reproducibles sobre diez grabaciones

```powershell
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\tools\prepare_tracking_benchmark.py
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\tools\benchmark_tracking.py prepare
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\tools\benchmark_tracking.py run --backend yolo --output "$env:LOCALAPPDATA\TennisLab\tracking_benchmark\results_yolo_reacquired"
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\tools\evaluate_tracking.py --score-only --results "$env:LOCALAPPDATA\TennisLab\tracking_benchmark\results_yolo_reacquired"
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\tools\benchmark_tracking.py report --output "$env:LOCALAPPDATA\TennisLab\tracking_benchmark\results_yolo_reacquired"
& "$env:LOCALAPPDATA\TennisLab\.venv\Scripts\python.exe" .\tools\benchmark_tracking.py verify --output "$env:LOCALAPPDATA\TennisLab\tracking_benchmark\results_yolo_reacquired"
```

La preparacion requiere el clip nocturno de la prueba real anterior,
con su ficha en `real_demo/night_source.json`. Los otros nueve se
descargan desde fuentes licenciadas. No se necesita ese material para
usar la aplicacion o ejecutar sus pruebas unitarias. Si no conservas
el clip nocturno, aporta una copia licenciada y su ficha antes de
ejecutar este benchmark concreto.

Para repetir el benchmark con fusion de escalas, cambia `--backend yolo`
por `--backend yolo_multiscale` y elige una carpeta `--output` distinta en
los cuatro comandos de ejecucion, puntuacion, informe y verificacion.
No mezcles resultados de motores distintos en una misma carpeta.

Se procesan **cuatro segundos de cada archivo distinto**, no diez
partidos completos. El conjunto incluye interiores, tomas nocturnas,
tierra batida, encuadres laterales, dobles y drones; algunos comparten
evento o pista, por lo que tampoco son diez dominios independientes.
Solo las referencias de suelo realmente revisables reciben una
calibracion. Los otros clips se analizan para pelota en imagen y
detecciones de personas, **sin proyectar jugadores ni contar esos
casos como exitos de pista**.

Cada clip se ejecuta en un proceso separado. Se conservan MP4 original
y analizado, JSON, tiempo real, tiempo CPU muestreado, pico de RSS del
arbol de procesos y bytes exportados. El muestreo de recursos es cada
50 ms, incluye FFmpeg y puede contar paginas compartidas dos veces.
Son mediciones de una pasada en un escritorio compartido, no un
laboratorio aislado. Los trabajos ya terminados se conservan; para
medir una revision distinta utiliza otro directorio de salida.
La accion `verify` decodifica los veinte MP4 completos y comprueba
fotogramas, dimensiones, timestamps y que los clips sin calibracion
no afirmen una proyeccion de jugadores.

`tracking_labels.json` contiene una pequena revision visual previa a
los overlays. Las coordenadas usan una imagen 640 x 360 y tolerancia
de 8 px. `null` significa **no evaluable**, no ausencia de pelota ni
acierto. El numero localizado entre los fotogramas visibles no es
precision global, F1 ni una certificacion. `evaluate_ball_model.py`
permite contrastar resoluciones 640/1280 del detector alternativo;
`evaluate_tracking.py` tambien compara un experimento de color y
movimiento, sin inferir que mas cobertura signifique mas calidad.
Las mismas etiquetas tambien se usaron para diagnosticar fallos de
asociacion; no se presentan como un conjunto de test ciego.

El directorio de salida incluye `VER_RESULTADOS.html`, `report.csv`,
`report.json` y `quality.json`. Los servicios externos de pago cuestan
0 EUR en estas ejecuciones locales. **No se ha medido electricidad**:
el informe separa un ejemplo a 30 W y 0,25 EUR/kWh del consumo real
desconocido y no incluye preparacion, descargas ni otros experimentos.

#### Resultado historico de la iteracion anterior (version 2)

En Windows 11, Ryzen 5 PRO 7530U (6 nucleos / 12 hilos), unos 14,8 GiB
de RAM utilizable y Python 3.14.7, con YOLO 640 y dos hilos de inferencia:

| Indicador | Resultado |
|---|---|
| Material | 10 archivos distintos, 4 s por archivo, 40 s en total |
| Tiempo de procesos, incluido arranque | 499,0 s en total; 42,8-68,6 s por clip |
| RSS pico del arbol por clip | 509-715 MiB, incluido FFmpeg |
| CPU muestreada, suma de los diez | 906,0 CPU-s |
| Exportaciones MP4/JSON/CSV y diagnostico | 31,7 MB decimales |
| Localizacion en la pequena revision visible | 13/15 a 8 px; 2 posiciones incorrectas |
| Pista / jugadores proyectados evaluables | 2 con calibracion parcial revisada; 8 sin proyeccion evaluable |
| Validacion de archivos | 20 MP4 decodificados completos, 2400 fotogramas |

Se revisaron 40 fotogramas; los otros 25 eran ambiguos para etiquetar
la pelota y no se cuentan como aciertos. La primera asociacion YOLO
solo localizaba 4/15: se quedaba enganchada a candidatos debiles de
pelotas de reserva y bloqueaba medidas nuevas mejores. Una prueba
de regresion reprodujo el bloqueo. Se acoto el sesgo temporal y
se permitio recuperar detecciones mas fuertes, sin cambiar el
comportamiento conservador de TrackNet. El resultado final **13/15
no es una precision general ni un test ciego**.

Sobre el mismo fragmento nocturno de 4 s, la comparacion completa fue
205,7 s con TrackNet frente a 50,5 s con YOLO, ambos con render:
aproximadamente 4,1 veces mas rapido. El pico RSS paso de 1452 a
715 MiB. No son pruebas repetidas aisladas de carga ajena.
En ese clip, la cobertura de registro de camara fue 92,7%, del
jugador cercano 92,7% y del lejano 82,9%; no se ha certificado
la precision metrica de la proyeccion.

### Viabilidad en un movil

La interfaz adaptable no ejecuta el motor Python dentro del telefono.
No se ha construido una app Android/iOS ni medido bateria, temperatura
o FPS en un dispositivo real. No se extrapola la velocidad de este
Ryzen a un procesador ARM.

Se uso el [comprobador oficial de ONNX Runtime](https://onnxruntime.ai/docs/tutorials/mobile/helpers/model-usability-checker.html)
para los modelos. Personas y pista tienen buen soporte estatico.
TrackNet y la copia de YOLO con entrada fija presentan fragmentacion;
el comprobador **no recomienda su aceleracion NNAPI/CoreML tal cual**.
Eso no prueba que sea imposible ejecutarlos en CPU movil ni da una
estimacion de latencia. La inferencia simbolica adicional tampoco
elimino las dos formas dinamicas internas de YOLO.

Una version nativa necesita exportacion y optimizacion especificas,
validacion de resultados tras cuantizar y mediciones sostenidas en
telefonos representativos. Otra arquitectura posible es procesar de
forma asincrona en un servidor, con autenticacion y consentimiento
de subida; este servidor local **no se expone a la red** para simularlo.
Consulta el [flujo oficial de desarrollo movil](https://onnxruntime.ai/docs/tutorials/mobile/).

El workflow de GitHub Actions permite ejecutar las pruebas unitarias bajo
demanda en Windows con Python 3.14, sin descargar pesos ni videos. No se
dispara automaticamente al subir codigo, para no consumir minutos de un
repositorio privado sin que lo decidas. Antes de activarlo, comprueba la
cuota y facturacion de tu cuenta. Las pruebas de inferencia real y de
navegador se ejecutan localmente, con modelos y grabaciones propios.

Las pruebas usan el video que ya proporcionaste, incluidos sus overlays, y casos sinteticos para comprobar errores. **No equivalen a un benchmark de precision con anotaciones humanas ni validan todos los moviles, pistas o iluminaciones.**

## Modelos, fuentes y licencias

No se ha entrenado aqui un modelo nuevo. Los pesos son de terceros y se preparan solo en el almacenamiento local:

- YOLOX-S de OpenCV Zoo, commit `48b10fa505aeb51fbd127fc60a5c660040e823b7`, licencia Apache-2.0. SHA-256 fijado en `setup_models.py`.
- Pista: [kjfk/tennis-court-keypoints-mirror](https://huggingface.co/kjfk/tennis-court-keypoints-mirror), revision `68342e926f5d012c79a02f8cc045bb75ca443ec8`, espejo de los pesos de `abdullahtarek/tennis_analysis`.
- Pelota: [kjfk/tracknet-tennis-ball-mirror](https://huggingface.co/kjfk/tracknet-tennis-ball-mirror), revision `1cd650376c405fed6ae5103bad704310a6e52dda`, espejo de los pesos de `yastrebksv/TrackNet`.

Las fichas de ambos mirrors declaran MIT. **Sus repositorios originales no incluyen un archivo LICENSE**, por lo que no se redistribuye su codigo. La procedencia y licencias declaradas se conservan en `models/sources.json` y `models/model_cards`. Revisa los derechos de los pesos antes de una redistribucion comercial; la etiqueta de un mirror no sustituye esa comprobacion.

La conversion usa carga restringida `weights_only=True`; no se ejecutan archivos Python descargados ni modelos proporcionados por el usuario.

Se descargaron tambien clips de Wikimedia Commons con licencias explicitas para evaluar limitaciones de encuadre. Sus metadatos se conservan en `fixtures/sources.json`, fuera de esta carpeta. Los encuadres laterales y de dobles no se han presentado como ejemplos de individuales exitosos.

Consulta tambien [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
Antes de compartir un video exportado, comprueba tus permisos sobre la
grabacion y conserva la atribucion que requiera su licencia.

## Que se publica en este repositorio

Solo codigo, interfaz, manifiestos de dependencias, pruebas y documentacion.
Se excluyen explicitamente grabaciones, resultados, bases de datos, pesos,
entornos virtuales, archivos `.env`, accesos directos con rutas personales e
informes privados de verificacion. No se publica una instancia del servidor
ni se configura ningun servicio de pago.
