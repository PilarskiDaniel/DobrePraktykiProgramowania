import cv2
import os
import time
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


def get_ground_truth():
    if not os.path.exists('annotations.xml'):
        return {}
    tree = ET.parse('annotations.xml')
    root = tree.getroot()
    gt_data = {}
    for image in root.findall('image'):
        name = image.get('name').strip()
        box = image.find('box')
        if box is not None:
            gt_text = "UNKNOWN"
            for attr in box.findall('attribute'):
                if attr.get('name') == 'plate number':
                    gt_text = attr.text.strip().upper() if attr.text else "UNKNOWN"
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            coords = [min(xtl, xbr), min(ytl, ybr), max(xtl, xbr), max(ytl, ybr)]
            gt_data[name] = {'text': gt_text, 'box': coords}
    return gt_data


def preprocess_for_ocr(img_crop):
    if img_crop.size == 0: return img_crop
    h, w = img_crop.shape[:2]
    if h < 40:
        scale = 40 / h
        img_crop = cv2.resize(img_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img_crop

def main():
    MODEL_PATH = r"runs/detect/train/weights/best.pt"
    TEST_DIR = 'datasets/val/images'

    if not os.path.exists(MODEL_PATH):
        for root, dirs, files in os.walk('runs'):
            if 'best.pt' in files:
                MODEL_PATH = os.path.join(root, 'best.pt')
                break

    if not os.path.exists(MODEL_PATH):
        print(f"Brak modelu best.pt! Sprawdź ścieżkę.")
        return

    print("Ładowanie modeli...")
    yolo_model = YOLO(MODEL_PATH)
    ocr = LicensePlateRecognizer('cct-xs-v1-global-model')
    gt_dict = get_ground_truth()

    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png'))]

    correct_ocr = 0
    total_checked = 0
    iou_scores = []

    cv2.namedWindow("Weryfikacja OCR", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Weryfikacja OCR", 1200, 800)

    print(f"{'PLIK':<15} | {'OCR':<12} | {'GT':<12} | {'WYNIK'}")
    print("-" * 60)

    for img_name in image_files:
        if img_name not in gt_dict:
            continue

        gt_info = gt_dict[img_name]
        img_path = os.path.join(TEST_DIR, img_name)
        img = cv2.imread(img_path)
        vis_img = img.copy()

        results = yolo_model(img, verbose=False)

        if len(results[0].boxes) > 0:
            total_checked += 1
            best_box = max(results[0].boxes, key=lambda x: x.conf)
            pred_box = best_box.xyxy[0].cpu().numpy()

            # IoU
            iou = calculate_iou(pred_box, gt_info['box'])
            iou_scores.append(iou)

            # Wycinanie (Crop)
            x1, y1, x2, y2 = map(int, pred_box)
            h_img, w_img = img.shape[:2]

            # Padding
            w_box = x2 - x1
            h_box = y2 - y1
            pad_x = int(w_box * 0.05)
            pad_y = int(h_box * 0.10)

            x1_p = max(0, x1 - pad_x)
            y1_p = max(0, y1 - pad_y)
            x2_p = min(w_img, x2 + pad_x)
            y2_p = min(h_img, y2 + pad_y)

            roi = img[y1_p:y2_p, x1_p:x2_p]

            pred_text = ""
            if roi.size > 0:
                roi_processed = preprocess_for_ocr(roi)
                try:
                    res = ocr.run(roi_processed)
                    if res:
                        pred_text = str(res[0]).upper()
                except Exception:
                    pred_text = "ERR"

            gt_text = gt_info['text']

            p_clean = "".join(filter(str.isalnum, pred_text))
            g_clean = "".join(filter(str.isalnum, gt_text))
            match = (p_clean == g_clean) and (g_clean != "")

            if match: correct_ocr += 1

            color = (0, 255, 0) if match else (0, 0, 255)

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)

            label = f"OCR: {p_clean} | GT: {g_clean}"

            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(vis_img, (x1, y1 - 35), (x1 + w_txt, y1), color, -1)

            cv2.putText(vis_img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Log w konsoli
            mark = '✅' if match else '❌'
            print(f"{img_name:<15} | {p_clean:<12} | {g_clean:<12} | {mark}")

            cv2.imshow("Weryfikacja OCR", vis_img)

            key = cv2.waitKey(0)
            if key == ord('q'):
                print("Przerwano przez użytkownika.")
                break
        else:
            print(f"{img_name}: BRAK DETEKCJI YOLO")
            cv2.putText(vis_img, "BRAK DETEKCJI YOLO", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Weryfikacja OCR", vis_img)
            if cv2.waitKey(0) == ord('q'): break

    cv2.destroyAllWindows()

    accuracy = (correct_ocr / total_checked * 100) if total_checked > 0 else 0
    print("\n" + "=" * 40)
    print(f"KONIEC. Skuteczność: {accuracy:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()