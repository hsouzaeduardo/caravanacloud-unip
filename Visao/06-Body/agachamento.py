import cv2
import mediapipe as mp
import numpy as np
import math

# Inicializar a detecção de pose do MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Função para calcular o ângulo entre três pontos
def calculate_angle(a, b, c):
    a = np.array(a) # Primeiro ponto
    b = np.array(b) # Segundo ponto (vértice)
    c = np.array(c) # Terceiro ponto
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Função para avaliar o agachamento
def squat_score(angle):
    if angle > 160:
        return "Pessimo"
    elif 140 < angle <= 140:
        return "Ruim"
    elif 90 < angle <= 120:
        return "Bom"
    else:
        return "Excelente"

# Captura de vídeo da câmera do notebook
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertendo imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Fazendo a detecção da pose
    results = pose.process(image)
    
    # Convertendo a imagem de volta para BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Desenhar as anotações da pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Coordenadas dos pontos do quadril, joelho e tornozelo
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculando o ângulo do joelho
        angle = calculate_angle(hip, knee, ankle)
        
        # Avaliando o agachamento
        evaluation = squat_score(angle)
        
        # Definindo a cor do texto com base na avaliação
        color = (0, 0, 255)  # Vermelho
        if evaluation in ["Bom", "Excelente"]:
            color = (0, 255, 0)  # Verde
        
        # Definindo o tamanho da fonte e posição
        font_scale = 3
        font_thickness = 6
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculando a posição central para o texto
        text_size = cv2.getTextSize(evaluation, font, font_scale, font_thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        
        # Exibindo o ângulo e a avaliação na imagem
        cv2.putText(image, str(angle), 
                    tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)), 
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
                    )
        cv2.putText(image, evaluation, 
                    (text_x, text_y), 
                    font, 3, color, 6, cv2.LINE_AA
                    )
        
    except:
        pass
    
    # Mostrando a imagem
    cv2.imshow('Squat Detection', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
