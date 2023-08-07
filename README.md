# Chartlytics-Backend
An application for the visually impaired that solves the problem of screen readers where they are unable to read the chart content from chart images, by generating a chart description for the input chart image uploaded on the application.

Our proposed solution supports data extraction from four types of charts:
<ol>
<li>Pie Charts
<li>Line Charts
<li>Horizontal Bar Charts
<li>Vertical Bar Charts
</ol>
The app includes a functionality that converts written text into spoken words, which leads the user through the app.
<h2>This repository contains: </h2>
<ul>
<li><a href="https://drive.google.com/drive/folders/1R5ZaGmnp6Oa1E19RV41tWHJypmLfcmL7">The classification model (MobileNet) used to classify the chart type</a>
<li>The object detection model (YOLOv5) weights used to identify chart components
<li>Data extraction code that is used to extract the raw data from the chart by using Rule-Based techniques.
</ul>

<h2> Chartlytics Dataset </h2> 
These datasets were generated by python scripts using Matplot Library.

They were used in the classification and the data extraction module alongside <a href="https://www.microsoft.com/en-us/research/project/figureqa-dataset/"> FigureQA </a> dataset.

<ul>
<li><a href="https://drive.google.com/drive/folders/1WewWAXXtW-fipTEfz9z9Lg6noEjf1ACd">Line Plots dataset</a>
<li><a href="https://drive.google.com/file/d/1q0xICah53D-U-uzb3qRnUMnQBdEdude9/view">Legend dataset</a>
</ul>

<h2> Results </h2>
<ul>
  <li>Our classification model excels at classifying the four chart types with 99.42% accuracy.
  <li>The Object detection model achieves 0.83-0.99 mAP in identifying different chart components.
  <li>The Data extraction module achieved 89.2%-100% accuracy.
  <li> Our chart description was rated as highly effective by majority of participants in our survey, with between 75% and 94% finding it to be accurate to the corresponding chart image.

  

