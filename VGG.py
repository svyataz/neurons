import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# 2. Загрузка и выбор датасета: 'cifar10' или 'cifar100'
DATASET = 'cifar10' # Измените на 'cifar100' при необходимости
if DATASET == 'cifar10':
 (x_train, y_train), (x_test, y_test) = cifar10.load_data()
 num_classes = 10
 classes = ['Автомобиль', 'Вертолёт', 'Птица', 'Кошка', 'Олень',
 'Собака', 'Лягушка', 'Лошадь', 'Корабль', 'Грузовик']
elif DATASET == 'cifar100':
 (x_train, y_train), (x_test, y_test) = cifar100.load_data()
 num_classes = 100
 # Если используете CIFAR-100, определите список классов или используйте стандартные метки
 # classes = [...]
else:
 raise ValueError("Неподдерживаемый набор данных. Используйте 'cifar10' или'cifar100'.")
# 3. Предобработка данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# Стандартизация данных
mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)
# One-hot encoding меток
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# 4. Создание tf.data.Dataset объектов для эффективной подачи данных
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=50000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
# 5. Определение упрощённой VGG-подобной модели
def build_simplified_vgg(input_shape=(32, 32, 3), num_classes=10):
 weight_decay = 1e-4
 model = models.Sequential()
 # Блок 1
 model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
 kernel_regularizer=regularizers.l2(weight_decay),
 input_shape=input_shape))
 model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
 kernel_regularizer=regularizers.l2(weight_decay)))
 model.add(layers.MaxPooling2D((2, 2)))
 model.add(layers.Dropout(0.25))
 # Блок 2
 model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
 kernel_regularizer=regularizers.l2(weight_decay)))
 model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
 kernel_regularizer=regularizers.l2(weight_decay)))
 model.add(layers.MaxPooling2D((2, 2)))
 model.add(layers.Dropout(0.25))
 # Блок 3
 model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu',
 kernel_regularizer=regularizers.l2(weight_decay)))
 model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu',
 kernel_regularizer=regularizers.l2(weight_decay)))
 model.add(layers.MaxPooling2D((2, 2)))
 model.add(layers.Dropout(0.25))
 # Полносвязные слои
 model.add(layers.Flatten())
 model.add(layers.Dense(512, activation='relu',
kernel_regularizer=regularizers.l2(weight_decay)))
 model.add(layers.Dropout(0.5))
 model.add(layers.Dense(num_classes, activation='softmax'))
 return model
# Построение модели
model = build_simplified_vgg(input_shape=x_train.shape[1:],
num_classes=num_classes)
model.summary()
# 6. Компиляция модели
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
 loss='categorical_crossentropy',
 metrics=['accuracy'])
# 7. Определение колбэков для ранней остановки и изменения скорости обучения
early_stopping = EarlyStopping(monitor='val_loss', patience=15,
restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
min_lr=1e-6, verbose=1)
# 8. Обучение модели
EPOCHS = 10 # Максимальное количество эпох, ранняя остановка может прекратить обучение раньше
history = model.fit(train_dataset,
 epochs=EPOCHS,
validation_data=test_dataset,
callbacks=[early_stopping, reduce_lr],
verbose=1)
# 9. Сохранение модели
model.save('trained_simplified_vgg.h5')
print("Модель успешно сохранена как 'trained_simplified_vgg.h5'")
# 10. Предсказание классов на тестовом наборе данных
predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
# 11. Функция для отображения нескольких предсказаний
def plot_predictions(x, y_true, y_pred, classes, num=25):
 plt.figure(figsize=(10, 10))
 for i in range(num):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(x[i])
  color = 'green' if y_pred[i] == y_true[i] else 'red'
  plt.xlabel(f"Pred: {classes[y_pred[i]]}\nTrue: {classes[y_true[i]]}", color=color)
 plt.tight_layout()
 plt.show()
# Отображение предсказаний
plot_predictions(x_test, true_classes, predicted_classes, classes, num=25)
# 12. Матрица путаницы и отчёт о классификации
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes,
yticklabels=classes)
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.title('Матрица путаницы')
plt.show()
print("Отчёт о классификации:")
print(classification_report(true_classes, predicted_classes,
target_names=classes))