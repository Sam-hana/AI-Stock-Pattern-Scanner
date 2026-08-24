#Use VGG style network
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    # 1. Define paths and parameters
    dataset_dir = "Dataset"
    batch_size = 16  #mini batch size of 16
    img_height = 64
    img_width = 64
    
    # 2. Load the dataset
    # Split the data: 80% for training, 20% for validation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Three types: W_Bottom, Head_Shoulder, No_Pattern
    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")
    
    # 3. Build the CNN Model
    model = tf.keras.Sequential([
        # Do the data augmentation
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        
        # Block 1: Conv2D with 32 filters, 3x3 kernel, ReLU activation
        # Because the image is only candlestick chart, no need filter as large as 64
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
    
        
        # Block 2: Conv2D with 64 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        
        # Flatten the tensor output from the convolutional layers
        tf.keras.layers.Flatten(),
    
        # The FC layer
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.6), 
        
        # Final Output Layer: Units should equal the number of classes. 
        # Use the appropriate activation for multi-class classification.
        tf.keras.layers.Dense(units=len(class_names), activation='softmax')
    ])
    
    # 4. Compile the model
    # EXERCISE: Use the standard optimizer for deep learning and the correct loss function.
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Add this to prevent all predicting to no_pattern
    weights_dict = {
        0: 1.15, 
        1: 1.0,   
        2: 1.15    
    }
    # 5. Train the model
    print("\nStarting model training...")
    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=weights_dict
    )
    
    model.save('stock_pattern_cnn.keras')
    print("\nModel saved as 'stock_pattern_cnn.keras'!")
if __name__=="__main__":
    main()