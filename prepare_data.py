import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

# Konfiguracja
PHOTOS_DIR = 'photos'
ANNOTATIONS_FILE = 'annotations.xml'
OUTPUT_DIR = 'datasets'

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

for path in ['train/images', 'train/labels', 'val/images', 'val/labels']:
    os.makedirs(os.path.join(OUTPUT_DIR, path), exist_ok=True)


def convert_xml_to_yolo():
    if not os.path.exists(ANNOTATIONS_FILE):
        print("Błąd: Brak pliku annotations.xml")
        return

    tree = ET.parse(ANNOTATIONS_FILE)
    root = tree.getroot()
    all_images = []

    for image in root.findall('image'):
        img_name = image.get('name')
        width_img = int(image.get('width'))
        height_img = int(image.get('height'))

        yolo_data = []
        for box in image.findall('box'):
            x1 = float(box.get('xtl'))
            y1 = float(box.get('ytl'))
            x2 = float(box.get('xbr'))
            y2 = float(box.get('ybr'))

            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)

            dw = 1. / width_img
            dh = 1. / height_img

            w = xmax - xmin
            h = ymax - ymin
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0

            # Normalizacja
            x_center *= dw
            w *= dw
            y_center *= dh
            h *= dh

            # Klasa 0
            yolo_data.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if yolo_data:
            all_images.append({'name': img_name, 'data': yolo_data})

    # Podział test 30% / train 70%
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.3, random_state=42)

    def save_set(dataset, folder):
        for item in dataset:
            src_path = os.path.join(PHOTOS_DIR, item['name'])
            if not os.path.exists(src_path):
                continue

            shutil.copy(src_path, os.path.join(OUTPUT_DIR, folder, 'images'))

            label_name = os.path.splitext(item['name'])[0] + '.txt'
            with open(os.path.join(OUTPUT_DIR, folder, 'labels', label_name), 'w') as f:
                f.write("\n".join(item['data']))

    save_set(train_imgs, 'train')
    save_set(val_imgs, 'val')

    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images
names:
  0: license-plate
"""
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    print(f"Dane gotowe. Train: {len(train_imgs)}, Val: {len(val_imgs)}")


if __name__ == "__main__":
    convert_xml_to_yolo()