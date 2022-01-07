import cv2
import numpy
from vehicle_detector import VehicleDetector

# vídeo a ser verificado
video = cv2.VideoCapture('bejamin.mkv')
# video = cv2.VideoCapture('santo-antonio.avi')


# carregamento das classes
arquivoDeClasses = "coco.names"
nomesDasClasses = []
with open(arquivoDeClasses, 'rt') as arquivo:
    nomesDasClasses = arquivo.read().rstrip('\n').split('\n')
print(nomesDasClasses)

# array de cores para delimitar as diferentes classes
cores = [[0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255]]

vehicleDetector = VehicleDetector()

regiaoDeInteresse = []


def desenharModelos(ids, scores, caixas):
    for (idDaClasse, score, caixa) in zip(ids, scores, caixas):
        cor = cores[int(idDaClasse) % len(cores)]

        label = f"{nomesDasClasses[idDaClasse[0]]}: {int(score*100)}%"

        cv2.rectangle(regiaoDeInteresse, caixa, cor, 2)
        cv2.putText(regiaoDeInteresse, label, (caixa[0], caixa[1]-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, cor)


def contarVeiculos(numeroDeVeiculos):
    cv2.putText(regiaoDeInteresse, str(numeroDeVeiculos), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0))


while True:
    ret, frame = video.read()
    regiaoDeInteresse = frame[120:400, 580:1080]

    ids, scores, caixas = vehicleDetector.detect_vehicles(
        regiaoDeInteresse)

    desenharModelos(ids, scores, caixas)
    contarVeiculos(len(ids))

    # Mostrar vídeo
    cv2.imshow('Image', regiaoDeInteresse)

    # Parar vídeo apertando 'q'
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
video.release()
