import cv2

faceCascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
users = [{'id': 0, 'name': 'Unknown', 'password': '000'}, {'id': 1, 'name': 'Cheep', 'password': 'shinosukayuki'}, 
    {'id': 2, 'name': 'Champ', 'password': 'champcheicheikodaiwa'}, {'id': 3, 'name': 'Peak', 'password': 'ilovecrypto'}, 
    {'id': 4, 'name': 'Tonnam', 'password': 'ineedsomesleep'}]
id_results = []
usernames = [user['name'] for user in users]

def display_info(user):
    print(f"""
    Name: {user['name']}
    """)

def identify(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    id = 0
    confidence = 100
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, confidence = clf.predict(gray[y: y + h, x: x + w])
        if confidence > 50:
            id = 0
    return img, id, round(100 - confidence)

def conclude(id_results):
    ids_appeared = {id_result[0]: 0 for id_result in id_results}
    if len(ids_appeared) == 1:
        return list(ids_appeared)[0]
    else:
        for id_result in id_results:
            ids_appeared[id_result[0]] += 1
        frequencies = list(ids_appeared.values())
        most_appeared = list(ids_appeared.keys())[frequencies.index(max(frequencies))]
        return most_appeared

username = input('Please input your username: ')
password = input('Please input your password: ')
if not username in usernames or username == 'Unknown':
    print('Invalid username! Try again.')
elif password != users[usernames.index(username)]['password']:
    print("Password doesn't match! Try again.")
else:
    cap = cv2.VideoCapture(0)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read('train/classifier.xml')

    for take in range(60):
        ret, frame = cap.read()
        frame, id, confidence = identify(frame, faceCascade, 1.1, 10, (0, 255, 0), clf)
        id_results.append([id, confidence])
        # print(f'take: {take}, id: {id}')
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    concluded_id = conclude(id_results)
    # print(f'concluded id = {concluded_id}')
    if concluded_id == usernames.index(username):
        print('Login successful\n')
        display_info(users[concluded_id])
    else:
        print('Authentication failed. Try again.')
        
    cv2.destroyAllWindows()