import cv2
import numpy

# vídeo a ser verificado
video = cv2.VideoCapture('bejamin.mkv')
# video = cv2.VideoCapture('santo-antonio.avi')


# constantes
whT = 320
limiarDeConfianca = 0.3         # em porcentagem
limiarDeSupressaoMaxima = 0.1   # em porcentagem
# limiarDeConfianca = 0.4         # em porcentagem
# limiarDeSupressaoMaxima = 0.15  # em porcentagem


# carregamento das classes
arquivoDeClasses = "coco.names"
nomesDasClasses = []
with open(arquivoDeClasses, 'rt') as arquivo:
    nomesDasClasses = arquivo.read().rstrip('\n').split('\n')
print(nomesDasClasses)

# array de cores para delimitar as diferentes classes
cores = numpy.random.uniform(0, 255, size=(len(nomesDasClasses), 3))

# Model Files
# configuracaoDosModelos = "yolov4.cfg"
# pesosDosModelos = "yolov4.weights"
configuracaoDosModelos = "yolov4-tiny.cfg"
pesosDosModelos = "yolov4-tiny.weights"

net = cv2.dnn.readNetFromDarknet(configuracaoDosModelos, pesosDosModelos)
# OpenCV como Backend. Não CPU
net.setPreferableBackend(cv2.dnn. DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def encontrarVeiculos(saidas, frame):
    altura, largura, cT = frame.shape

    caixas = []
    idsDasClasses = []
    confiabilidade = []

    for saida in saidas:
        for resultados in saida:
            # Pagar os cinco primeiros valores da analise da rede. (cx, cy, largura, altura, confianca da classe)
            scores = resultados[5:]
            # Pegar a classificacao
            idDaClasse = numpy.argmax(scores)
            # Pegar a confianca da classificacao
            confianca = scores[idDaClasse]
            # Filtrar nossa classificacao
            if confianca > limiarDeConfianca:
                larguraDoQuadro, alturaDoQuadro = int(
                    resultados[2]*largura), int(resultados[3]*altura)

                xDoQuadro, yDoQuadro = int(
                    (resultados[0]*largura)-larguraDoQuadro / 2), int((resultados[1]*altura)-alturaDoQuadro/2)

                caixas.append(
                    [xDoQuadro, yDoQuadro, larguraDoQuadro, alturaDoQuadro])

                idsDasClasses.append(idDaClasse)
                confiabilidade.append(float(confianca))

    # evitar varias caixas. Supressao Maxima
    indices = cv2.dnn.NMSBoxes(
        caixas, confiabilidade, limiarDeConfianca, limiarDeSupressaoMaxima)
    print(len(indices))

    # Criação das caixas
    for i in indices:
        i = i[0]
        if nomesDasClasses[idsDasClasses[i]] in ['car', 'motorcycle', 'bus', 'truck', 'bicycle']:
            caixa = caixas[i]
            x, y, largura, altura = caixa[0], caixa[1], caixa[2], caixa[3]
            # print(x,y,w,h)
            cv2.rectangle(frame, (x, y), (x+largura, y+altura),
                          cores[idsDasClasses[i]], 2)
            cv2.putText(frame, f'{nomesDasClasses[idsDasClasses[i]].upper()} {int(confiabilidade[i]*100)}%',
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cores[idsDasClasses[i]], 2)


while True:
    ret, frame = video.read()
    regiaoDeInteresse = frame[120:400, 580:1080]
    # regiaoDeInteresse = frame[120:650, 730:1180]

    blob = cv2.dnn.blobFromImage(
        regiaoDeInteresse, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    nomeDasLayers = net.getLayerNames()  # Camadas da rede CNN treinada

    # Não usamos o valor zero das camadas de saida por isso a subtracao com 1.
    # Pegar o nome das camadas de saida -> print(outputNames)
    nomesDeSaida = [(nomeDasLayers[i[0] - 1])
                    for i in net.getUnconnectedOutLayers()]  # Na saida temos três tipos. Por isso um vetor com tres posicoes em net.getUnconnectedOutLayers()

    # Saida das camadas da rede com a confianca predita
    saidas = net.forward(nomesDeSaida)
    encontrarVeiculos(saidas, regiaoDeInteresse)

    # Mostrar vídeo
    cv2.imshow('Image', regiaoDeInteresse)

    # Parar vídeo apertando 'q'
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
video.release()
