import cv2
import mediapipe as mp
import time
import math
import pyttsx3

#Webcam & MediaPipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#State tracking
pTime = 0
prev_letter = ""
letter_hold_count = 0
text = ""
HOLD_FRAMES = 15

#Helpers
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def speak(text_to_say):
    try:
        print(f"[SPEAK] {text_to_say}")
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text_to_say)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS ERROR] {e}")

def detect_letter(lm, img_shape):
    h, w, _ = img_shape
    coords = lambda i: (int(lm[i].x * w), int(lm[i].y * h))

    thumb_tip = coords(4)
    index_tip = coords(8)
    middle_tip = coords(12)
    ring_tip = coords(16)
    pinky_tip = coords(20)
    index_mcp = coords(5)
    middle_mcp = coords(9)
    ring_mcp = coords(13)
    pinky_mcp = coords(17)

    # A
    if (
        index_tip[1] > index_mcp[1]
        and middle_tip[1] > middle_mcp[1]
        and ring_tip[1] > ring_mcp[1]
        and pinky_tip[1] > pinky_mcp[1]
    ):
        return "A"

    # B
    if (
        index_tip[1] < index_mcp[1]
        and middle_tip[1] < middle_mcp[1]
        and ring_tip[1] < ring_mcp[1]
        and pinky_tip[1] < pinky_mcp[1]
    ):
        return "B"

    # L
    if (
        index_tip[1] < index_mcp[1]
        and middle_tip[1] > middle_mcp[1]
        and ring_tip[1] > ring_mcp[1]
        and pinky_tip[1] > pinky_mcp[1]
        and abs(index_tip[0] - thumb_tip[0]) > 40
    ):
        return "L"

    # O
    if distance(thumb_tip, index_tip) < 40:
        return "O"

    # V
    if (
        index_tip[1] < index_mcp[1]
        and middle_tip[1] < middle_mcp[1]
        and ring_tip[1] > ring_mcp[1]
        and pinky_tip[1] > pinky_mcp[1]
        and distance(index_tip, middle_tip) > 40
    ):
        return "V"

    return ""

#Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    letter = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            letter = detect_letter(lm, img.shape)

    # Handle letter holding
    if letter == prev_letter:
        letter_hold_count += 1
    else:
        letter_hold_count = 0

    if letter and letter_hold_count == HOLD_FRAMES:
        text += letter
        print(f"[TEXT] {text}")

    prev_letter = letter

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # UI
    cv2.putText(img, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Letter: {letter}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    cv2.putText(img, f"Text: {text}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow("ASL A-B-L-V-O (Speak on W)", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        text = ""
        print("[TEXT RESET]")
    elif key == ord('w'):
        if text:
            speak(text)

cap.release()
cv2.destroyAllWindows()

