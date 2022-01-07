import cv2
import numpy

# Arquivo dos modelos
configuracaoDosModelos = "yolov4.cfg"
pesosDosModelos = "yolov4.weights"
# configuracaoDosModelos = "yolov4-tiny.cfg"
# pesosDosModelos = "yolov4-tiny.weights"

confiabilidadeMinima = 0.5


class VehicleDetector:

    def __init__(self):
        # Load Network
        net = cv2.dnn.readNetFromDarknet(
            configuracaoDosModelos, pesosDosModelos)

        # OpenCV como Backend. Não CPU
        net.setPreferableBackend(cv2.dnn. DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.modelo = cv2.dnn_DetectionModel(net)
        self.modelo.setInputParams(size=(416, 416), scale=1 / 255)

        # Classes de veículos
        self.classesPermitidas = [2, 3, 5, 7]

        # Atualização inicial
        self.buffer = 29

        # Definição das caixas
        self.caixas = []
        self.ids = []
        self.taxaDeCerteza = []

    def detect_vehicles(self, img):
        # Detect Objects

        # Adiciona contagem ao buffer
        self.buffer = self.buffer+1

        # Verificação de uma vez por segundo (todo: pegar parâmetro fps do vídeo posteriormente)
        if(self.buffer == 30):
            # Zera as caixas atuais
            self.caixas = []
            self.ids = []
            self.taxaDeCerteza = []

            idsDasClasses, scores, boxes = self.modelo.detect(
                img, nmsThreshold=0.4)
            for idDaClasse, score, caixa in zip(idsDasClasses, scores, boxes):
                if score < confiabilidadeMinima:
                    continue

                if idDaClasse in self.classesPermitidas:
                    self.caixas.append(caixa)
                    self.ids.append(idDaClasse)
                    self.taxaDeCerteza.append(score)

            # zera o buffer atual
            self.buffer = 0

        return self.ids, self.taxaDeCerteza, self.caixas
