from ultralytics import YOLO
import torch
import os


def start_training():
    # Konfiguracja urządzenia
    if torch.cuda.is_available():
        device = '0'
        print(f"Znaleziono GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Nie znaleziono GPU - trening odbędzie się na CPU.")

    # Załadowanie modelu
    model = YOLO('yolov8n.pt')

    print("Rozpoczynam trening...")

    # rening z ustawieniami "Safe Mode" dla Windows
    try:
        results = model.train(
            data='data.yaml',
            epochs=30,
            imgsz=640,
            device=device,
            batch=8,
            workers=1,

            project='runs/detect',
            name='train',
            exist_ok=True,
            plots=True
        )
        print("Trening zakończony sukcesem!")
        print(f"Model zapisany w: {os.path.abspath('runs/detect/train/weights/best.pt')}")

    except Exception as e:
        print(f"\nBŁĄD KRYTYCZNY PODCZAS TRENINGU:\n{e}")
        print("\nSugestia: Jeśli błąd dotyczy pamięci (CUDA OOM), zmniejsz 'batch' na 4.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    start_training()