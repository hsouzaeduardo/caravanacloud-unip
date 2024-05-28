import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar o MediaPipe para a detecção de pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis do jogo
game_started = False
game_over = False
movement_detected = False
round_active = False
start_time = 0
movement_threshold = 20  # Sensibilidade ao movimento

# Função para detectar movimento
def detect_movement(prev_frame, current_frame):
    diff = cv2.absdiff(prev_frame, current_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) > 0

# Loop principal do jogo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Verificar se a imagem tem 3 canais (RGB)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Converter a imagem para RGB para a detecção de pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Desenhar as anotações de pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Checar o movimento
        if game_started and not game_over:
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if 'prev_frame_gray' in locals():
                movement_detected = detect_movement(prev_frame_gray, current_frame_gray)
            prev_frame_gray = current_frame_gray

            if round_active:
                if movement_detected:
                    game_over = True
                    print("Game Over! Você se moveu!")
                elif time.time() - start_time > 5:
                    round_active = False
                    print("Parado, você pode se mover agora.")
            else:
                if time.time() - start_time > 3:
                    round_active = True
                    start_time = time.time()
                    print("Batatinha Frita 1, 2, 3! Não se mova!")

        # Mostrar instruções iniciais
        if not game_started:
            cv2.putText(frame, "Pressione 's' para iniciar o jogo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Verificar entrada do usuário
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            game_started = True
            start_time = time.time()
        elif key == ord('q'):
            break

        # Mostrar a imagem da câmera
        cv2.imshow('Round Six - Batatinha Frita 1, 2, 3', frame)
    else:
        print("Imagem capturada não possui 3 canais (RGB).")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
