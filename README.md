# Breast Ultrasound Cancer Segmentation

U-Net модель для сегментации опухолей на УЗИ молочной железы.

## 📦 Используемые технологии

- TensorFlow / Keras
- NumPy, OpenCV
- Matplotlib
- Jupyter Notebook

## 📁 Датасет

Используется датасет: [Kaggle - Breast Ultrasound Images](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

Помести его в папку `Dataset_BUSI_with_GT/` рядом с ноутбуком.

## 🚀 Как запустить

1. Установи зависимости:

```bash
   pip install -r requirements.txt
```

2. Запусти ноутбук:

```bash
   jupyter notebook
```

3. Открой и выполни `breast_cancer_unet.ipynb`

## 🧠 Дообучение

Можно дообучать модель, используя больше размеченных изображений. Просто дозагрузи новые данные и запусти `model.fit(...)`.

## 💾 Сохранение и использование модели

- Модель сохраняется в файл:

```python
model.save('BreastCancerSegmentor.h5')
```
