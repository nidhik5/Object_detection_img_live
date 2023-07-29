import cv2
import numpy as np

print(cv2.__version__)

net = cv2.dnn.readNet("C:\\Users\\Nidhi\\Desktop\\yolov3.weights", "C:\\Users\\Nidhi\\Desktop\\yolov3.cfg")
classes = []
with open("C:\\Users\\Nidhi\\Desktop\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

num_classes = len(classes)
print("Number of Classes:", num_classes)

cap = cv2.VideoCapture(0)  # 0 for the default webcam, or provide the index for a specific camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # For debugging, inspect the output shape
    #print("Output Shapes:", [out.shape for out in outs])

    boxes = []
    confidences = []
    class_ids = []

    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            print(class_id)
            confidence = scores[class_id]
            

            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.5)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
