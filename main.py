import cv2
import os
import time
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    # Minimalne wymagania
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    # Normalizacja
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    # Wynik ważony
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    # Skala 2.0 - 5.0
    grade = 2.0 + 3.0 * score
    grade = min(5.0, max(2.0, grade))
    return round(grade * 2) / 2

def calculate_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

def get_ground_truth():
    if not os.path.exists('annotations.xml'):
        print("Błąd: Brak pliku annotations.xml")
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
            xtl, ytl, xbr, ybr = float(box.get('xtl')), float(box.get('ytl')), float(box.get('xbr')), float(box.get('ybr'))
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
    MODEL_PATH = r"C:\Users\Daniel\PycharmProjects\OCR\runs\detect\runs\detect\train\weights\best.pt"
    TEST_DIR = 'datasets/val/images'

    print("Ładowanie modeli...")
    yolo_model = YOLO(MODEL_PATH)
    ocr = LicensePlateRecognizer('cct-xs-v1-global-model')
    gt_dict = get_ground_truth()

    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        print("Brak zdjęć w folderze!")
        return

    iou_scores, correct_ocr, total_checked = [], 0, 0
    print(f"Rozpoczynam test na {len(image_files)} zdjęciach...")
    start_time = time.time()

    print(f"\n{'PLIK':<15} | {'OCR':<12} | {'GT':<12} | {'IoU':<6} | {'WYNIK'}")
    print("-" * 75)

    for img_name in image_files:
        if img_name not in gt_dict: continue
        gt_info = gt_dict[img_name]
        img = cv2.imread(os.path.join(TEST_DIR, img_name))
        results = yolo_model(img, verbose=False)

        if len(results[0].boxes) > 0:
            total_checked += 1
            best_box = max(results[0].boxes, key=lambda x: x.conf)
            pred_box = best_box.xyxy[0].cpu().numpy()
            iou = calculate_iou(pred_box, gt_info['box'])
            iou_scores.append(iou)

            x1, y1, x2, y2 = map(int, pred_box)
            h_img, w_img = img.shape[:2]
            w_box, h_box = x2 - x1, y2 - y1
            pad_x, pad_y = int(w_box * 0.05), int(h_box * 0.10)
            roi = img[max(0, y1-pad_y):min(h_img, y2+pad_y), max(0, x1-pad_x):min(w_img, x2+pad_x)]

            pred_text = ""
            if roi.size > 0:
                roi_p = preprocess_for_ocr(roi)
                try:
                    res = ocr.run(roi_p)
                    pred_text = str(res[0]).upper() if res else ""
                except: pred_text = "ERR"

            p_clean = "".join(filter(str.isalnum, pred_text))
            g_clean = "".join(filter(str.isalnum, gt_info['text']))
            match = (p_clean == g_clean) and (g_clean != "")
            if match: correct_ocr += 1
            print(f"{img_name:<15} | {p_clean:<12} | {g_clean:<12} | {iou:.2f} | {'✅' if match else '❌'}")

    total_time = time.time() - start_time
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    accuracy = (correct_ocr / total_checked * 100) if total_checked > 0 else 0
    time_per_100 = (total_time / len(image_files)) * 100 if image_files else 0
    final_grade = calculate_final_grade(accuracy, time_per_100)

    print("\n" + "=" * 50)
    print("RAPORT KOŃCOWY:")
    print(f"Średnie IoU:      {avg_iou:.4f}")
    print(f"Dokładność OCR:   {accuracy:.2f}%")
    print(f"Poprawne:         {correct_ocr} / {total_checked}")
    print(f"Czas na 100 zdj:  {time_per_100:.2f} s")
    print(f"OCENA KOŃCOWA:    {final_grade}")
    print("=" * 50)

if __name__ == "__main__":
    main()