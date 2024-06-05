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

# Função para avaliar a pose de fisiculturismo "double biceps"
def double_biceps_score(left_angle, right_angle):
    # Ângulo ideal para "double biceps" (exemplo, pode ajustar conforme necessário)
    ideal_angle = 90
    
    # Calculando a diferença absoluta dos ângulos
    left_diff = abs(ideal_angle - left_angle)
    right_diff = abs(ideal_angle - right_angle)
    
    # Calculando a nota baseada nas diferenças (quanto menor a diferença, maior a nota)
    score = max(0, 10 - (left_diff + right_diff) / 2 / 9)
    
    return score

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
        
        # Coordenadas dos pontos do ombro, cotovelo e punho para ambos os lados
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Calculando os ângulos dos cotovelos
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Avaliando a pose "double biceps"
        score = double_biceps_score(left_angle, right_angle)
        
        # Definindo a cor do texto como preto
        color = (0, 0, 0)  # Preto
        
        # Definindo o tamanho da fonte e posição
        font_scale = 3
        font_thickness = 6
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculando a posição para o texto na direita
        text = f"Score: {score:.2f}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = image.shape[1] - text_size[0] - 20  # 20 pixels de margem
        text_y = text_size[1] + 20  # 20 pixels de margem superior
        
        # Exibindo a nota na imagem
        cv2.putText(image, text, 
                    (text_x, text_y), 
                    font, font_scale, color, font_thickness, cv2.LINE_AA
                    )
        
    except:
        pass
    
    # Mostrando a imagem
    cv2.imshow('Double Biceps Pose Detection', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
