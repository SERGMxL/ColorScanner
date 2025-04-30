import cv2
import numpy as np

def detectar_colores_y_personas():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Detector de colores y personas iniciado. Presiona 'q' para salir.")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_resultado = frame.copy()
        frame_personas = frame.copy()
        vision_rgb = np.zeros_like(frame)  # Imagen negra

        # Diccionario de colores (nombre: [rango HSV, color de dibujo])
        colores = {
            "Negro": ([0, 0, 0], [180, 255, 50], (255, 255, 255)),  # ahora se resalta en blanco
            "Rojo1": ([0, 100, 100], [10, 255, 255], (0, 0, 255)),
            "Rojo2": ([160, 100, 100], [180, 255, 255], (0, 0, 255)),
            "Verde": ([40, 70, 70], [80, 255, 255], (0, 255, 0)),
            "Azul": ([100, 150, 0], [140, 255, 255], (255, 0, 0))
        }

        for nombre, valor in colores.items():
            if "Rojo" in nombre:
                continue

            lower = np.array(valor[0])
            upper = np.array(valor[1])
            color_bgr = valor[2]

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)

            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contornos:
                area = cv2.contourArea(cnt)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(cnt)  
                    cv2.rectangle(frame_resultado, (x, y), (x+w, y+h), color_bgr, 2)
                    cv2.putText(frame_resultado, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            vision_rgb[np.where(mask != 0)] = color_bgr

        # Procesar el ROJO (dos rangos)
        rojo_mask1 = cv2.inRange(hsv, np.array(colores["Rojo1"][0]), np.array(colores["Rojo1"][1]))
        rojo_mask2 = cv2.inRange(hsv, np.array(colores["Rojo2"][0]), np.array(colores["Rojo2"][1]))
        rojo_mask = cv2.bitwise_or(rojo_mask1, rojo_mask2)
        rojo_mask = cv2.erode(rojo_mask, None, iterations=1)
        rojo_mask = cv2.dilate(rojo_mask, None, iterations=2)

        contornos_rojo, _ = cv2.findContours(rojo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contornos_rojo:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                color_rojo = colores["Rojo1"][2]
                cv2.rectangle(frame_resultado, (x, y), (x+w, y+h), color_rojo, 2)
                cv2.putText(frame_resultado, "Rojo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rojo, 2)

        vision_rgb[np.where(rojo_mask != 0)] = colores["Rojo1"][2]

        # Detección de personas
        personas, _ = hog.detectMultiScale(frame_personas, winStride=(8, 8))
        for (x, y, w, h) in personas:
            cv2.rectangle(frame_personas, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame_personas, 'Persona', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Mostrar ventanas
        cv2.imshow('Imagen de camara (con detección de personas)', frame_personas)
        cv2.imshow('Vista del robot (detección de colores)', frame_resultado)
        cv2.imshow('Visión RGB del robot', vision_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_colores_y_personas()
