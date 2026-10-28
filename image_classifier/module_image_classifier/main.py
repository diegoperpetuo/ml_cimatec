import json
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import typer

from module_image_classifier.config import FIGURES_DIR, HISTORY_PATH, MODELS_DIR

app = typer.Typer(help="Classificador de imagens CIFAR-10.")

MODEL_PATH = MODELS_DIR / "model.keras"

CLASS_NAMES = [
    "avião", "automóvel", "pássaro", "gato", "cervo",
    "cachorro", "sapo", "cavalo", "navio", "caminhão",
]


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def load_data():
    """Carrega o CIFAR-10, normaliza para [0, 1] e achata os rótulos."""
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    logger.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")
    return (X_train, y_train), (X_test, y_test)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def build_model(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """CNN simples com data augmentation, BatchNorm e Dropout."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            # Data augmentation (ativo só durante o treino)
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            # Bloco 1
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Bloco 2
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Bloco 3
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            # Classificador
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_history(history: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="treino")
    axes[0].plot(history["val_loss"], label="validação")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="treino")
    axes[1].plot(history["val_accuracy"], label="validação")
    axes[1].set_title("Acurácia")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Curvas de treino salvas em {output_path}")


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Matriz de confusão salva em {output_path}")


# --------------------------------------------------------------------------- #
# Comandos
# --------------------------------------------------------------------------- #
@app.command()
def train(
    epochs: int = typer.Option(30, help="Número máximo de épocas."),
    batch_size: int = typer.Option(64, help="Tamanho do batch."),
    validation_split: float = typer.Option(0.1, help="Fração do treino usada para validação."),
):
    """Treina o modelo e salva o melhor checkpoint em models/model.keras."""
    (X_train, y_train), _ = load_data()

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(CLASS_NAMES))
    model.summary(print_fn=logger.info)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
    )

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    logger.info(f"Histórico salvo em {HISTORY_PATH}")

    plot_history(hist, FIGURES_DIR / "training_curves.png")
    logger.success(f"Treino concluído. Modelo salvo em {MODEL_PATH}")


@app.command()
def evaluate():
    """Avalia o modelo salvo no conjunto de teste."""
    if not MODEL_PATH.exists():
        logger.error(f"Modelo não encontrado em {MODEL_PATH}. Rode `train` primeiro.")
        raise typer.Exit(code=1)

    _, (X_test, y_test) = load_data()
    model = keras.models.load_model(MODEL_PATH)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Loss no teste: {loss:.4f} | Acurácia no teste: {acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_test, y_pred), 1)

    for i, name in enumerate(CLASS_NAMES):
        class_acc = cm[i, i] / cm[i].sum()
        logger.info(f"  {name:<10} acurácia: {class_acc:.3f}")

    plot_confusion_matrix(cm, FIGURES_DIR / "confusion_matrix.png")
    logger.success("Avaliação concluída.")


@app.command()
def predict(
    image_path: Path = typer.Argument(..., exists=True, readable=True, help="Caminho da imagem."),
):
    """Classifica uma imagem qualquer (redimensionada para 32x32)."""
    if not MODEL_PATH.exists():
        logger.error(f"Modelo não encontrado em {MODEL_PATH}. Rode `train` primeiro.")
        raise typer.Exit(code=1)

    model = keras.models.load_model(MODEL_PATH)

    img = keras.utils.load_img(image_path, target_size=(32, 32))
    x = keras.utils.img_to_array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]

    logger.info(f"Imagem: {image_path}")
    for rank, idx in enumerate(top3, start=1):
        logger.info(f"  {rank}. {CLASS_NAMES[idx]:<10} {probs[idx] * 100:.1f}%")
    logger.success(f"Previsão: {CLASS_NAMES[top3[0]]}")


if __name__ == "__main__":
    app()