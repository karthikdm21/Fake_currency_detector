<h1> Indian Currency Counterfeit Detection System Using Image Processing </h1>

<h2>Overview</h2>
<p>
This project focuses on verifying the authenticity of Indian currency notes by implementing a system that processes the image of a currency note and determines whether it is genuine or fake. The system leverages advanced image processing and computer vision techniques, built entirely in Python using Jupyter Notebooks. OpenCV is utilized for image analysis, while Tkinter is used to create an interactive GUI for user input and result display.
</p>

<h2>Technologies and Libraries Used</h2>
<ul>
  <li>OpenCV - For image processing</li>
  <li>Tkinter - For GUI development</li>
  <li>Numpy - For numerical operations</li>
  <li>Matplotlib - For visualization</li>
  <li>Jupyter Notebook - For development and execution</li>
</ul>

<h2>Dataset Information</h2>
<p>
The system currently supports only Indian currency notes of denominations ₹500 and ₹2000. The dataset includes both real and counterfeit currency images, which have been manually collected from various sources. This dataset plays a crucial role in training and testing the model to identify key features that distinguish real notes from fake ones.
</p>

<h2>Project Structure</h2>
<ul>
  <li>Dataset Folder: Contains authentic and fake currency note images.</li>
  <li>Fake Notes Folder: Includes fake ₹500 and ₹2000 currency images for testing.</li>
  <li>500_testing.ipynb: Processes images of ₹500 currency notes.</li>
  <li>2000_testing.ipynb: Processes images of ₹2000 currency notes.</li>
  <li>controller.ipynb: The main controller notebook that handles input and generates output.</li>
  <li>gui_1.ipynb: GUI for uploading the input image.</li>
  <li>gui_2.ipynb: GUI for displaying the final result.</li>
  <li>FAKE_CURRENCY_DETECTOR_REPORT: Contains a detailed project report, including methodology, dataset structure, and results.</li>
</ul>

<h2>How It Works</h2>
<ol>
  <li>Clone the repository to your local machine.</li>
  <li>Open controller.ipynb in Jupyter Notebook.</li>
  <li>Run all the cells:<br><em>Cell &gt; Run All</em></li>
  <li>A GUI window will pop up:<br>
    <ul>
      <li>Click 'Select an Image' to upload a currency note image.</li>
      <li>Choose the denomination (₹500 or ₹2000).</li>
      <li>Click 'Submit'.</li>
    </ul>
  </li>
  <li>A second window will appear with a progress bar to show the processing status.</li>
  <li>The final result window will display:
    <ul>
      <li>The input image</li>
      <li>The number of authentic features detected</li>
      <li>SSIM (Structural Similarity Index) score for each feature</li>
      <li>Pass/Fail status for each feature</li>
    </ul>
  </li>
</ol>

<h2>Output Highlights</h2>
<ul>
  <li>Input currency note image</li>
  <li>Number of authentic features detected</li>
  <li>SSIM similarity score for each feature</li>
  <li>Pass/Fail status of each feature</li>
  <li>Final authenticity result</li>
</ul>

<h2>Key Improvements and Unique Features</h2>
<ul>
  <li>✅ Real-time currency analysis using advanced image processing techniques.</li>
  <li>✅ Supports both ₹500 and ₹2000 denomination notes.</li>
  <li>✅ GUI-based system for easy image selection and result visualization.</li>
  <li>✅ SSIM-based feature similarity analysis for accurate detection.</li>
  <li>✅ Visual representation of detected features and authenticity score.</li>
  <li>✅ Custom dataset built specifically for Indian currency notes.</li>
</ul>

<h2>Possible Future Enhancements</h2>
<ul>
  <li>Adding support for more Indian currency denominations.</li>
  <li>Enhancing accuracy by implementing deep learning models.</li>
  <li>Building a mobile-based application for on-the-go currency verification.</li>
  <li>Integrating OCR (Optical Character Recognition) for serial number verification.</li>
</ul>

<h2>Conclusion</h2>
<p>
This Fake Currency Detection System efficiently identifies counterfeit Indian currency notes using image processing techniques. With the help of OpenCV and Tkinter, the system accurately analyzes key features and provides a detailed report on the authenticity of the note. This project not only enhances the accuracy of counterfeit detection but also paves the way for future advancements in currency verification systems.
</p>

<p><strong>"Say goodbye to fake notes and stay one step ahead with this intelligent detection system!"</strong></p>
