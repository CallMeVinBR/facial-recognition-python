import cv2
import numpy as np
import json
import os

classificador_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificador_olhos = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numero_amostras = 25

# Usado para dar um formato aos arquivos de fotos e
# atribuir os nomes dos arquivos corretamente:
# pessoa.{id}.{numero_foto}.jpg
id = int(input("Digite o seu ID: "))
nome = input("Digite o seu nome: ")

dados_pessoa = {
    f"{id}": {
        "nome": nome
    }
}

arquivo_json = "pessoas.json"

with open(arquivo_json, "r") as infile:
    try:
        dados_existentes = json.load(infile)
    except json.JSONDecodeError:
        dados_existentes = []

dados_existentes.append(dados_pessoa)

with open(arquivo_json, "w") as outfile:
    json.dump(dados_existentes, outfile, indent=4)

largura, altura = 150, 150
print("Inicializando a camera...\n")
print(" CONTROLES ".center(25, "="))
print("\n[C]\tCapturar foto\n[R]\tRecapturar uma foto\n[1]\tParar execução")

while True:
    conectado, imagem = camera.read()
    if not conectado:
        break
    # Processamento da imagem para uma escala de preto e branco
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador_face.detectMultiScale(imagem_cinza,
                                                      scaleFactor=1.5,
                                                      minSize=(125,125))
    
    # l, a = largura e altura
    # x, y = onde começa uma face
    for x, y, l, a in faces_detectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 255, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
        regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhos_detectados = classificador_olhos.detectMultiScale(regiao_cinza_olho)
        
        if len(olhos_detectados) == 0:
            cv2.putText(regiao, "Sem olhos", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            for ox, oy, ol, oa in olhos_detectados:
                # Isolando a região do olho
                roi_eye = regiao_cinza_olho[oy:oy + oa, ox:ox + ol]
                
                # Calculando a proporção de pixels brancos
                _, thresh = cv2.threshold(roi_eye, 50, 255, cv2.THRESH_BINARY)
                pixels_brancos = np.sum(thresh == 255)
                total_pixels = np.size(thresh)
                ratio = pixels_brancos / total_pixels
                
                # Verificando se o olho está aberto ou fechado
                if ratio > 0.2:
                    cv2.putText(regiao, "Aberto", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
        
            # Verificando se a tecla C (captura) foi pressionada
        tecla = cv2.waitKey(1)
        if tecla == ord('c'):
            if amostra > numero_amostras:
                break
            else:
                # As imagens devem ser tiradas em um ambiente iluminado
                # para que essa condição seja verdadeira
                imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
                if np.average(imagem_face) > 110:
                    cv2.imwrite(f"fotos/pessoa.{id}.{amostra}.jpg", imagem_face) # Criando o arquivo na pasta...
                    print(f"Foto {amostra} capturada com sucesso!")
                    amostra += 1 # Partindo para a próxima amostra com o incremento...
                else:
                    print("Iluminação insuficiente no rosto. Tente novamente.")
        elif tecla == ord('1'):
            camera.release()
            cv2.destroyAllWindows()
            break
        elif tecla == ord('r'): # Recaptura
            numero_foto = int(input("Digite o numero da foto a ser recapturada, de 1 a 25: "))
            while numero_foto <= 0 or numero_foto > 25:
                numero_foto = int(input("Digite o numero da foto a ser recapturada, de 1 a 25: "))
                
            print(f"Recapturando foto {numero_foto}...")
            imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
            if np.average(imagem_face) > 110:
                cv2.imwrite(f"fotos/pessoa.{id}.{numero_foto}.jpg", imagem_face)
            else:
                print("Iluminação insuficiente no rosto. Tente novamente.")
                
    cv2.imshow("Face", imagem)

print("Camera desligada.")
camera.release()
cv2.destroyAllWindows()