from ultralytics import YOLO
import cv2


def detectar_animais(imagem_path):
    # carregar modelo YOLO pré-treinado
    model = YOLO("yolov8n.pt")

    # rodar detecção
    resultados = model(imagem_path)

    # pegar imagem
    imagem = cv2.imread(imagem_path)

    for resultado in resultados:
        boxes = resultado.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confianca = float(box.conf[0])
            classe = int(box.cls[0])

            label = f"{model.names[classe]} {confianca:.2f}"

            # desenhar bounding box
            cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                imagem,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("Deteccao", imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detectar_animais("imagem_teste.jpg")