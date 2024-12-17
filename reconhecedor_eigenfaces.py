import cv2
import json

camera = cv2.VideoCapture(0)
detector_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer.create()
reconhecedor.read("classificadorEigen.yml")
largura, altura = 150, 150
font = cv2.FONT_HERSHEY_SIMPLEX

with open('pessoas.json', 'r') as openfile:
    pessoa_dados = json.load(openfile)

while True:
    conectado, imagem = camera.read()
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = detector_face.detectMultiScale(imagem_cinza, 
                                                      scaleFactor=1.5,
                                                      minSize=(30,30))
    for x, y, l, a in faces_detectadas:
        imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 255, 255), 2)
        id, confianca = reconhecedor.predict(imagem_face)
        pessoa = None
        for dado in pessoa_dados:
            if str(id) in dado:
                pessoa = dado[str(id)]["nome"]
                break
        
        if pessoa:
            cv2.putText(imagem, pessoa, (x, y + (a + 30)), font, 2, (0, 255, 0), 2)
        else:
            cv2.putText(imagem, "Desconhecido", (x, y + (a + 30)), font, 2, (0, 255, 0))
    
    cv2.imshow("Face", imagem)
    tecla = cv2.waitKey(1)
    if tecla == ord('1'):
        break
    
camera.release()
cv2.destroyAllWindows()