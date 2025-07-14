# Task 1: Edge AI Prototype

**Tools: TensorFlow Lite, Colab simulation*

**Goal* : 
***Train a lightweight image classification model (recognizing recyclable items). The model should be able to classify images into recyclable or non-recyclable.***

    **This Project* ::
        * On this edge ai solution for a smart assistive technology, a model is trained on how to make decisions on next action steps for a user, who we'll assume is using an assistive device, such a wheelchare, or blind.
        * The model uses data from realtime camera(or in future sensors), to detects objects ahead, takes a picture of the object, analyze and predict what is.
        * It then calculates the objects distance from the user and advices them to either stop moving or turn to avoid maybe collisions.
        * For this device now, should be able to assist a blind person for instance, to navigate with confidence from place to place, with the help of an edge ai assistive device toavoid obstacles on their way and enable faster movement.

***Convert the model to TensorFlow Lite and test it on a sample dataset. ***

    * This conversion phase to TensorFlow Lite model, is to enable the models deployment to an edge device like a Raspberry Pi, on actual devices.
    * The script loads MobileNetV2, converts it to TensorFlow Lite, and saves the TFLite model.

***Explain how Edge AI benefits real-time applications***

* Processing data directly localy on the device (edge) eliminates the need to send data to the cloud and wait for a response. This enables instant decision-making, which is critical for such assistive devices
* Improved Privacy:
Sensitive data (such as images or audio) is processed locally, reducing the risk of data breaches and protecting user privacy.
* Reliability:
Edge AI systems can operate even without a stable internet connection, ensuring continuous service in remote or mobile environments.

***Results***
* After running the model, the script loads MobileNetV2, converts it to TensorFlow Lite, and saves the TFLite model.
* It captures an image from your webcam, preprocesses it, and prepares it for inference.
* The TFLite interpreter runs inference on the image.
* The predictions are decoded to readable labels.
* A simulated distance measurement is combined with the object detection result to provide user guidancea like alerts a user to stop, or take a turn.

**Note* :: 

**Refer to code for the model and steps*

**The  screenshots are available from live testing*