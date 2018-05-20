from keras import callbacks

#https://keras.io/ja/callbacks/

def cb():
#    pl = callbacks.ProgbarLogger(count_mode='steps')
    history = callbacks.History()
    ch = callbacks.ModelCheckpoint('./weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                               monitor='val_loss',
                               verbose=0,
                               save_best_only=False,
                               save_weights_only=False,
                               mode='auto',
                               period=1)
    es = callbacks.EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=3,
                                verbose=0,
                                mode='auto')
    cb=[history, ch, es]

#学習率の変更をしたいならこれ
#keras.callbacks.LearningRateScheduler(schedule, verbose=0)

#テンソルボードに記述したいならこれ
#keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

#評価値の改善が止まった時に学習率を減らします．
#モデルは訓練が停滞した時に学習率を2〜10で割ることで恩恵を受けることがあります． このコールバックは評価値を監視し，'patience'で指定されたエポック数の間改善が見られなかった場合，学習率を減らします．
#keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    return cb, history
