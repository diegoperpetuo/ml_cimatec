from matplotlib import pyplot as plt

from image_classifier.module_image_classifier.dataset import X_train, y_train, class_names

# Exemplos da base CIFAR-10

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.suptitle("Exemplos da base CIFAR-10.")
plt.tight_layout()
plt.show()

# Acurácia e Loss

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label="Treino")
plt.plot(history.history['val_accuracy'], label="Validação")
plt.title("Acurácia por época")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Treino")
plt.plot(history.history['val_loss'], label="Validação")
plt.title("Loss por época")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Previsões em imagens de teste

pred_probs = model.predict(X_test[:9])
pred_labels = np.argmax(pred_probs, axis=1)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[i])
    real = class_names[y_test[i]]
    pred = class_names[pred_labels[i]]
    plt.title(f"Real: {real}\nPredição: {pred}")
    plt.axis('off')
plt.suptitle("Previsões em imagens do conjunto de teste")
plt.tight_layout()
plt.show()