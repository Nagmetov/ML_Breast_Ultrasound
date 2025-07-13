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

2. Установи Jupyter Notebook

```bash
   pip install jupyter
```

3. Запусти ноутбук:

```bash
   jupyter notebook
```

4. Открой и выполни `./model_learn/segmentation-model-for-breast-cancer.ipynb`

## 🧠 Дообучение

Можно дообучать модель, используя больше размеченных изображений. Просто дозагрузи новые данные и запусти `model.fit(...)`.

## 💾 Сохранение и использование модели

- Модель сохраняется в файл:

```python
model.save('BreastCancerSegmentor.h5')
```

Чтобы загрузить:

```python
from tensorflow.keras.models import load_model
model = load_model('BreastCancerSegmentor.h5')
```

# Результаты

| Входное изображение       | Истинная маска         | Предсказанная маска      |
| ------------------------- | ---------------------- | ------------------------ |
| ![](results/0_input.png)  | ![](results/0_gt.png)  | ![](results/0_pred.png)  |
| ![](results/4_input.png)  | ![](results/4_gt.png)  | ![](results/4_pred.png)  |
| ![](results/6_input.png)  | ![](results/6_gt.png)  | ![](results/6_pred.png)  |
| ![](results/8_input.png)  | ![](results/8_gt.png)  | ![](results/8_pred.png)  |
| ![](results/10_input.png) | ![](results/10_gt.png) | ![](results/10_pred.png) |
| ![](results/12_input.png) | ![](results/12_gt.png) | ![](results/12_pred.png) |
| ![](results/14_input.png) | ![](results/14_gt.png) | ![](results/14_pred.png) |
