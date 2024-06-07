class NN(tf.keras.Model):
    def __init__(self, model_name):
        '''
        Init the structure, when you init the structure in main.py, it will plot a summary of structure.
        my initial structure is that
                Input_img
                base_model
            GlobalAveragePooling2D
                Flatten
               Dropout 0.2
                 Dense 50
               Dropout 0.2
        
        Dense 10            Dense 10
        Dense 1(angle)     Dense 1(speed)

        '''
        super(NN, self).__init__()
        print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('Model: ', model_name)
        if model_name == 'ResNet50V2':
            self.base_model = tf.keras.applications.resnet_v2.ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3),
                input_tensor=None)
        elif model_name == 'ResNet101V2':
            self.base_model = tf.keras.applications.resnet_v2.ResNet101V2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
        elif model_name == 'ResNet50':
            self.base_model = tf.keras.applications.resnet50.ResNet50(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3)
            )
        elif model_name == 'VGG19':
            self.base_model = tf.keras.applications.vgg19.VGG19(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3)
            )
        elif model_name == 'ResNet152V2':
            self.base_model = tf.keras.applications.resnet_v2.ResNet152V2(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3)
            )
        elif model_name == 'VGG16':
            self.base_model = tf.keras.applications.vgg16.VGG16(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3))
        elif model_name == 'DenseNet201':
            self.base_model = tf.keras.applications.densenet.DenseNet201(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3),
            )
        elif model_name == 'DenseNet169':
            self.base_model = tf.keras.applications.densenet.DenseNet169(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3),
            )
        elif model_name == 'DenseNet121':
            self.base_model = tf.keras.applications.densenet.DenseNet121(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3),
            )
        elif model_name == 'EfficientNetV2S':
            self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3))
        elif model_name == 'EfficientNetV2M':
            self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3))
        elif model_name == 'EfficientNetV2L':
            self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3))
        else:
            raise ValueError("Invalid model_name. Supported options are 'ResNet50V2'.")

        '''
        This base_model has 195 layers, I frozen all of them, means will not train the base layer.
        If you want to fine-tune the pre-trained model, you can use the follow code, 
        that means fine-tune the layers which is over 190, others will be frozen.
        for i, layer in enumerate(self.base_model.layers):
            if i > 190:
                layer.trainable = True
            else:
                layer.trainable = False
        '''
        for i, layer in enumerate(self.base_model.layers):
            layer.trainable = True

        print('ALL layers: ', len(self.base_model.layers))  # Print the number of layers in base model.
        self.conv_reduce_channels = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')
        self.bn_reduce_channels = tf.keras.layers.BatchNormalization()
        self.relu_reduce_channels = tf.keras.layers.Activation('relu')
        self.conv_reduce_channels2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        # self.AveragePooling = tf.keras.layers.GlobalAveragePooling2D()
        # self.flatten = tf.keras.layers.Flatten()
        # self.Dropout_2 = tf.keras.layers.Dropout(0.5)
        # angle branch
        self.angle_branch = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear')])
        # speed branch
        self.speed_branch = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear')])
        self.build([None, 224, 224, 3])
        self.summary()

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.conv_reduce_channels(x)
        x = self.bn_reduce_channels(x)
        x = self.relu_reduce_channels(x)
        x = self.conv_reduce_channels2(x)
        output_1 = self.angle_branch(x)
        output_2 = self.speed_branch(x)
        return output_1, output_2

    # Read img and preprocess
    def preprocess_image(self, image_path, augment=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        if augment:
            image = tf.image.random_brightness(image, max_delta=0.2)  # Random brightness adjustment
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast adjustment
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Random saturation adjustment
            # image = tf.image.random_crop(image,(220,220,3))

        image = tf.image.resize(image, [224, 224]) / 255.0
        print("Data Process!")
        return image

    def load_data(self, root_path, img_path, csv_path, batch_size=16):
        # load csv
        labels_df = pd.read_csv(os.path.join(root_path, csv_path))

        # path process
        labels_df['image_path'] = labels_df['image_id'].apply(lambda x: os.path.join(root_path, img_path, f"{x}.png"))
        labels_df = labels_df[labels_df['image_path'].apply(os.path.exists)]
        # split into train dataset and validation dataset

        # Split data into two parts based on index
        original_datasets = labels_df[labels_df.index <= 13794]
        flip_data = labels_df[labels_df.index > 13794]
        train_df, val_df = train_test_split(original_datasets, test_size=0.2, random_state=6)
        # Combine remaining data with training set
        train_df = pd.concat([train_df, flip_data])
        # to tensor type
        train_img_dataset = tf.data.Dataset.from_tensor_slices((
            train_df['image_path'].values
        ))
        val_img_dataset = tf.data.Dataset.from_tensor_slices((
            val_df['image_path'].values
        ))
        train_label_dataset = tf.data.Dataset.from_tensor_slices((
            train_df['angle'].values,
            train_df['speed'].values
        ))
        val_label_dataset = tf.data.Dataset.from_tensor_slices((
            val_df['angle'].values,
            val_df['speed'].values
        ))

        train_img_dataset = train_img_dataset.map(lambda x: self.preprocess_image(x, augment=True),
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        val_img_dataset = val_img_dataset.map(lambda x: self.preprocess_image(x, augment=False),
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # zip img and label
        train_dataset = tf.data.Dataset.zip((train_img_dataset, train_label_dataset))
        val_dataset = tf.data.Dataset.zip((val_img_dataset, val_label_dataset))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_df) * 2)
        return train_dataset.batch(batch_size), val_dataset.batch(batch_size)

    def training(self, train_dataset, val_dataset, epochs, trained_model, model_save_path):
        if trained_model is not None:
            self.load_weights(trained_model)
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                     loss={'output_1': 'mse', 'output_2': 'mse'},
                     metrics=['mse'])
        self.build([None, 224, 224, 3])

        log_dir = log_path + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True)
        earlystopping_callback = EarlyStopping(patience=15, restore_best_weights=True)
        history_callback = History()
        start = time.time()
        history = self.fit(train_dataset,
                           epochs=epochs,
                           validation_data=val_dataset,
                           callbacks=[checkpoint_callback, tensorboard_callback, earlystopping_callback, history_callback])

        end = time.time()
        self.save(model_save_path, save_format='tf')
        print('Time: {:.2f} minutes'.format((end - start) / 60))

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(history.history['output_1_loss'], label='Training Angle Loss')
        plt.plot(history.history['val_output_1_loss'], label='Validation Angle Loss')
        plt.title('Angle Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(history.history['output_2_loss'], label='Training Speed Loss')
        plt.plot(history.history['val_output_2_loss'], label='Validation Speed Loss')
        plt.title('Speed Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(loss_img)
        return history

    def predict_model(self, trained_model_path, image_path, output_path):
        # Load already trained model
        
        label_interval = 0.0625
        num_labels = 17
        # Minimum value of the label
        min_label = 0.0
        
        # Label range for the predicted values
        label_range = [min_label + i * label_interval for i in range(num_labels)]
        self.load_weights(trained_model_path)
        # Open the CSV file, ready to write data
        with open(output_path, 'w', newline='') as csvfile:
            # Define the CSV writer
            writer = csv.writer(csvfile)
            # Write the CSV header
            writer.writerow(['image_id', 'angle', 'speed'])

            # Traverse the images in the folder and make predictions for each image
            for image_name in sorted(os.listdir(image_path)):
                img_path = os.path.join(image_path, image_name)
                # Load and preprocess the image
                # Read the image file and resize it
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                # Convert the image to a NumPy array and scale the pixel values to the [0, 1] range
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                # Reshape the image array to the model input format
                img_input = tf.expand_dims(img_array, axis=0)

                # Use the trained model to make predictions
                prediction = self.predict(img_input)
                
                # Process the prediction results
                angle = prediction[0][0][0]  # Extract value from 2D array
                speed = prediction[1][0][0] 
                if speed >= 0.6:
                    speed = 1
                elif speed <= 0.4:
                    speed = 0
                else:
                    speed = speed

                if angle >= 1:
                    angle = 1
                elif angle <= 0:
                    angle = 0
                else:
                    angle = angle

                # Round to the appropriate precision
                angle = np.round(angle, 4)
                speed = np.round(speed, 4)
                image_id = image_name.split('.')[0]

                writer.writerow([image_id, angle, speed])
        print('Finish!')
