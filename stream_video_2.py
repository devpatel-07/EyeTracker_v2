import cv2, zmq

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

sock = zmq.Context().socket(zmq.PUB)
sock.setsockopt(zmq.SNDHWM, 1)
sock.bind("tcp://*:5556 ")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    sock.send(jpg.tobytes())