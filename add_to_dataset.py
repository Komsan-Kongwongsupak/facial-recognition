from venv import create
import cv2

faceCascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")


def create_dataset(img, id, img_id):
    cv2.imwrite('dataset/pic.' + str(id) + "." + str(img_id) + ".jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        coords = [x, y, w, h]
    return img, coords

def detect(img, faceCascade, img_id, eyeCascade=None, noseCascade=None, mouthCascade=None):
    img, coords = draw_boundary(img, faceCascade, 1.1, 10, (0, 255, 0))
    if len(coords) == 4:
        id = 4 # the label of the photo
        result = img[coords[1]: coords[1] + coords[3], coords[0]: coords[0] + coords[2]]
        create_dataset(result, id, img_id)
    
    return img

# import from the camera
img_id = 1 # the number of the next photo from the last existing one; e.g. If the last number of the photo is 10, the img_id value is 11.
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = detect(frame, faceCascade, img_id)
    cv2.imshow('frame', frame)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()