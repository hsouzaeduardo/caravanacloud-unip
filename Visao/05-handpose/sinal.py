import cv2
import mediapipe as mp
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import os
import threading
import time

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Substitua com sua chave de API e ponto de extremidade do Azure
subscription_key = os.getenv('subscription_key')
service_region = os.getenv('service_region')

# Inicializar MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Inicializar o serviço de Text-to-Speech do Azure
speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=service_region)
speech_config.speech_synthesis_voice_name = "pt-BR-AntonioNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

def is_peace_sign(landmarks):
    # Verificar se os dedos indicador e médio estão levantados e os outros dedos estão abaixados
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    is_index_up = index_finger_tip.y < index_finger_mcp.y
    is_middle_up = middle_finger_tip.y < middle_finger_mcp.y
    is_ring_down = ring_finger_tip.y > ring_finger_mcp.y
    is_pinky_down = pinky_tip.y > pinky_mcp.y

    return is_index_up and is_middle_up and is_ring_down and is_pinky_down

def is_rock_sign(landmarks):
    # Verificar se os dedos indicador e mínimo estão levantados e os outros dedos estão abaixados
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    is_index_up = index_finger_tip.y < index_finger_mcp.y
    is_middle_down = middle_finger_tip.y > middle_finger_mcp.y
    is_ring_down = ring_finger_tip.y > ring_finger_mcp.y
    is_pinky_up = pinky_tip.y < pinky_mcp.y

    return is_index_up and is_middle_down and is_ring_down and is_pinky_up

def speak_text(text):
    speech_synthesizer.speak_text_async(text)

# Variável para controlar se o sinal de rock foi detectado
rock_sign_detected = False
last_rock_sign_time = 0

# Variável para controlar se o sinal de paz foi detectado
peace_sign_detected = False
last_peace_sign_time = 0

# Loop principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter a imagem para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem e detectar mãos
    results = hands.process(rgb_frame)

    # Desenhar as anotações de mão
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            current_time = time.time()
            
            # Verificar se o gesto de paz foi feito
            if is_peace_sign(hand_landmarks.landmark):
                cv2.putText(frame, "Sinal de Paz Detectado", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if current_time - last_peace_sign_time > 2:  # 2 segundos de intervalo
                    threading.Thread(target=speak_text, args=("Eu vôs dou a minha Paz",)).start()
                    last_peace_sign_time = current_time
                
            elif is_rock_sign(hand_landmarks.landmark):
                cv2.putText(frame, "Sinal de Rock Detectado", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if current_time - last_rock_sign_time > 2:  # 2 segundos de intervalo
                    threading.Thread(target=speak_text, args=("SIM isso é Rock in Roll baby !!",)).start()
                    last_rock_sign_time = current_time
                
            else:
                cv2.putText(frame, "Mostre o Sinal de Paz", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                peace_sign_detected = False

    # Mostrar a imagem da câmera
    cv2.imshow('Hand Pose - Peace Sign Detection', frame)

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
