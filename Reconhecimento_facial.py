import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read('recognizer/trainningData.yml')

fonte = cv2.FONT_HERSHEY_COMPLEX

def usuarios(id):
    int(id)
    if id == 7:
        return 'Marcelo Soares'
    if id == 5:
        return 'Lange Soares'
    else:
        return 'Id Nao Reconhecido'

def face_detector(frame, size=0.5):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = face_classifier.detectMultiScale(cinza, 1.3, 5)
    if rostos == ():
        return frame, []

    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (300, 300))

        return frame, roi

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    imagem, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            ID = result[0]
            chance = int((1 - (result[1]) / 300) * 100)
            display_string = str(chance) + '% Chance de Ser Um Usuario'
            cv2.putText(imagem, display_string, (30, 50), fonte, 1, (0, 255, 255), 4)

            if chance > 80:
                cv2.putText(imagem, "Desbloquado", (200, 450), fonte, 1, (0, 255, 0), 4)
                cv2.putText(imagem, usuarios(ID), (200, 300), fonte, 1, (0, 255, 0), 4)
                cv2.imshow('Face Scanner', imagem)

            else:
                cv2.putText(imagem, "Bloqueado", (200, 450), fonte, 1, (0, 0, 255), 4)
                cv2.imshow('Face Scanner', imagem)

    except:
        cv2.putText(imagem, "Rosto Sem Reconhecimento", (75, 450), fonte, 1, (0, 0, 255), 4)
        cv2.imshow('Face Scanner', imagem)
        pass

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
