Indian Currency Counterfeit Detection System Using Image Processing
Overview
This project focuses on verifying the authenticity of Indian currency notes by implementing a system that processes the image of a currency note and determines whether it is genuine or fake. The system leverages advanced image processing and computer vision techniques, built entirely in Python using Jupyter Notebooks. OpenCV is utilized for image analysis, while Tkinter is used to create an interactive GUI for user input and result display.

Technologies and Libraries Used
OpenCV - For image processing
Tkinter - For GUI development
Numpy - For numerical operations
Matplotlib - For visualization
Jupyter Notebook - For development and execution
Dataset Information
The system currently supports only Indian currency notes of denominations ₹500 and ₹2000. The dataset includes both real and counterfeit currency images, which have been manually collected from various sources. This dataset plays a crucial role in training and testing the model to identify key features that distinguish real notes from fake ones.

Project Structure
Dataset Folder: Contains authentic and fake currency note images.
Fake Notes Folder: Includes fake ₹500 and ₹2000 currency images for testing.
500_testing.ipynb: Processes images of ₹500 currency notes.
2000_testing.ipynb: Processes images of ₹2000 currency notes.
controller.ipynb: The main controller notebook that handles input and generates output.
gui_1.ipynb: GUI for uploading the input image.
gui_2.ipynb: GUI for displaying the final result.
FAKE_CURRENCY_DETECTOR_REPORT: Contains a detailed project report, including methodology, dataset structure, and results.
How It Works
Clone the repository to your local machine.
Open controller.ipynb in Jupyter Notebook.
Run all the cells:
Cell > Run All
A GUI window will pop up:
Click 'Select an Image' to upload a currency note image.
Choose the denomination (₹500 or ₹2000).
Click 'Submit'.
A second window will appear with a progress bar to show the processing status.
The final result window will display:
The input image
The number of authentic features detected
SSIM (Structural Similarity Index) score for each feature
Pass/Fail status for each feature
Output Highlights
Input currency note image
Number of authentic features detected
SSIM similarity score for each feature
Pass/Fail status of each feature
Final authenticity result
Key Improvements and Unique Features
✅ Real-time currency analysis using advanced image processing techniques.
✅ Supports both ₹500 and ₹2000 denomination notes.
✅ GUI-based system for easy image selection and result visualization.
✅ SSIM-based feature similarity analysis for accurate detection.
✅ Visual representation of detected features and authenticity score.
✅ Custom dataset built specifically for Indian currency notes.

Possible Future Enhancements
Adding support for more Indian currency denominations.
Enhancing accuracy by implementing deep learning models.
Building a mobile-based application for on-the-go currency verification.
Integrating OCR (Optical Character Recognition) for serial number verification.
Conclusion
This Fake Currency Detection System efficiently identifies counterfeit Indian currency notes using image processing techniques. With the help of OpenCV and Tkinter, the system accurately analyzes key features and provides a detailed report on the authenticity of the note. This project not only enhances the accuracy of counterfeit detection but also paves the way for future advancements in currency verification systems.

"Say goodbye to fake notes and stay one step ahead with this intelligent detection system!"
