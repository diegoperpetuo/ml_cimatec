from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3, # Se a val_loss não melhorar por 3 épocas consecutivas, o treinamento para.
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping]
)