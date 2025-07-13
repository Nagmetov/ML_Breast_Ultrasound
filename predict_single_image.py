import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys
import os

# Загрузка модели
model = load_model('./models/BreastCancerSegmentor.h5')

def predict_and_save(image_path, save_path='output.png', threshold=0.5):
    # Чтение изображения
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Преобразуем размер
    img_resized = cv2.resize(img, (256, 256))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Предсказание
    pred = model.predict(img_input)[0, :, :, 0]
    binary_mask = (pred > threshold).astype(np.uint8) * 255

    # Морфологическая очистка
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Визуализация и сохранение
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Predicted Mask")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Результат сохранён в {save_path}")

# === Точка входа ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Укажите путь к изображению: python predict_single_image.py path/to/image.png")
    else:
        image_path = sys.argv[1]
        predict_and_save(image_path)
