Computer vision project SVU 

This work was done by students :

1 - mahmoud mohamed nagi 

This project focuses on developing a computer vision system capable of identifying and classifying different types of Egyptian coins. The primary objective is to create a model that can accurately recognize the coin's category based on its image.

The project utilizes deep learning techniques, specifically the convolutional neural network (CNN) algorithm, implemented using popular libraries such as NumPy, TensorFlow, Keras, and OpenCV (cv2).

The CNN algorithm is a powerful deep learning architecture designed to process and analyze image data effectively. It consists of multiple convolutional layers that extract relevant features from the input images, followed by pooling layers that downsample the feature maps, and fully connected layers that perform the final classification.

The project pipeline involves several steps:

1. **Data Preprocessing**: A dataset of Egyptian coin images is collected and preprocessed. This includes resizing the images to a consistent size, converting them to a suitable format (e.g., grayscale or RGB), and splitting the dataset into training and validation sets.

2. **Model Architecture**: A CNN model is designed and implemented using libraries like TensorFlow and Keras. The architecture typically consists of multiple convolutional layers, pooling layers, and fully connected layers. The specific configuration, such as the number of layers, filter sizes, and activation functions, is tailored to the Egyptian coin classification task.

3. **Model Training**: The preprocessed dataset is fed into the CNN model for training. During this phase, the model learns to recognize patterns and features associated with different coin categories. Techniques like data augmentation (e.g., flipping, rotating, or scaling images) can be employed to increase the diversity of the training data and improve model generalization.

4. **Model Evaluation**: The trained CNN model is evaluated on a separate validation set to assess its performance. Metrics such as accuracy, precision, recall, and F1-score are commonly used to measure the model's effectiveness in classifying Egyptian coins correctly.

5. **Prediction and Visualization**: Once the model achieves satisfactory performance, it can be deployed for real-world application. When a new Egyptian coin image is provided as input, the model processes the image through the learned CNN architecture and outputs the predicted coin category. This prediction can be displayed to the user, along with additional information or visualizations if desired.

Throughout the project, libraries like NumPy are used for numerical computations, while OpenCV (cv2) is employed for image processing tasks, such as loading, resizing, and preprocessing the coin images.

By leveraging the power of CNNs and deep learning techniques, this project aims to build an accurate and reliable system for Egyptian coin identification, facilitating research, documentation, and preservation of historical artifacts.
