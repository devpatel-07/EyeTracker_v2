import cv2, zmq, numpy as np

sock = zmq.Context().socket(zmq.SUB)
sock.setsockopt(zmq.CONFLATE, 1)          # keep only the newest frame
sock.setsockopt_string(zmq.SUBSCRIBE, "")
sock.connect("tcp://172.17.90.232:5555")

while True:
    frame = cv2.imdecode(np.frombuffer(sock.recv(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("stream", frame)
    if cv2.waitKey(1) == 27:
        break