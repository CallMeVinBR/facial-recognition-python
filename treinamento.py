import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer.create()
fisherface = cv2.face.FisherFaceRecognizer.create()
lbph = cv2.face.LBPHFaceRecognizer.create()

def getImagemPorId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
        # cv2.imshow("Face", imagemFace)
        # cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemPorId()

print("Treinando...")

# No treinamento, passamos as faces e os respecivos IDs
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado.")