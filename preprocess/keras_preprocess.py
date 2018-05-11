from keras.preprocessing.image import ImageDataGenerator

a = ImageDataGenerator(featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    zca_epsilon=1e-06,
                    rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    brightness_range=None,
                    shear_range=0.0,
                    zoom_range=0.0,
                    channel_shift_range=0.0,
                    fill_mode='nearest',
                    cval=0.0,
                    horizontal_flip=False,
                    vertical_flip=False,
                    rescale=None,
                    preprocessing_function=None,
                    data_format=None,
                    validation_split=0.0)

##最終的にはjsonファイルから取り出すようにしたいね。


train_datagen = ImageDataGenerator(**data_gen_args_train)
test_datagen = ImageDataGenerator(**data_gen_args_test)

#これもjsonファイルで定義すると楽かもね。
train_generator_args = dict{
    directory, target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    follow_links=False,
    subset=None,
    interpolation='nearest')
}
train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size,
            color_mode,
            classes,
            class_mode,
            batch_size,
            save_to_dir,
            seed,
            save_to_dir


)
