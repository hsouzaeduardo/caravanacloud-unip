import cv2

# Carregar um modelo pré-treinado (por exemplo, MobileNet SSD)
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt.txt", 
    "MobileNetSSD_deploy.caffemodel"
)

# Definir as classes de objetos que o modelo pode detectar
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def count_people(frame):
    # Obter as dimensões do frame
    (h, w) = frame.shape[:2]
    # Converter o frame para um blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Passar o blob pelo modelo e obter as detecções
    net.setInput(blob)
    detections = net.forward()
    
    count = 0
    # Loop através das detecções
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filtrar detecções fracas
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            # Verificar se a classe detectada é uma pessoa
            if CLASSES[idx] == "person":
                count += 1
    return count

# Capturar vídeo da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Contar o número de pessoas no frame atual
    num_people = count_people(frame)
    print(f"Number of people: {num_people}")

    # Mostrar o frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()