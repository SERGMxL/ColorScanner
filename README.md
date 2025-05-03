# ColorScanner

# Detección de Colores y Personas con OpenCV

Este proyecto en Python permite detectar ciertos colores (rojo, verde, azul y negro) en tiempo real a través de la cámara web, así como identificar la presencia de personas mediante el uso del descriptor HOG (Histogram of Oriented Gradients) de OpenCV.

## 🎯 Funcionalidades

- 📸 Captura de video en tiempo real desde la cámara.
- 🎨 Detección de colores específicos en formato HSV:
  - Rojo (dos rangos)
  - Verde
  - Azul
  - Negro
- 🚶‍♂️ Detección de personas utilizando `cv2.HOGDescriptor`.
- 🖼 Visualización en tres ventanas:
  - Cámara con detección de personas
  - Detección de colores
  - Visión RGB del robot (en negro con colores resaltados)

## 🧰 Tecnologías utilizadas

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## 🔧 Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
