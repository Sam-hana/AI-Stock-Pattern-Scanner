# AI Stock Pattern Scanner

An automated post-market scanning tool that leverages a Convolutional Neural Network (CNN) to identify technical chart patterns (e.g., W-Bottom, Head & Shoulders) in S&P 500 stocks. 

[View the Live Daily Report Here](https://Sam-hana.github.io/AI-Stock-Pattern-Scanner/)

## Overview & Features

This project streamlines the daily routine of scanning hundreds of stock charts. By executing a single script, the system automatically fetches financial data,
processes it through a computer vision model, and highlights actionable trading setups.

* Automated Data Pipeline: Scrapes the S&P 500 ticker list and downloads the latest 60 days of historical data via yfinance.
* Deep Learning Vision: Uses a custom-trained VGG-style CNN model (TensorFlow/Keras) to classify candlestick charts into specific patterns.
* Dynamic HTML Reporting: Automatically generates a clean, responsive web report using Python, HTML, and CSS, deployed seamlessly via GitHub Pages.

## Tech Stack

* Machine Learning: TensorFlow, Keras, NumPy
* Data Processing: Pandas, yfinance, requests
* Visualization & Frontend: mplfinance, HTML, CSS

## Project Structure

* `generate_report.py`: The main controller. It calls the scanner and generates the index.html report.
* `scanner.py`: The core logic module. Handles data fetching, K-line chart generation, image preprocessing, and model prediction.
* `train_model.py`: The training script. Loads the dataset, applies data augmentation, and trains the VGG-style CNN.
* `stock_pattern_cnn.keras`: The saved weights and architecture of the trained model.
* `index.html`: The final output of a responsive daily report for identified stock patterns.

## Technical Challenge: Overcoming Class Imbalance

During the initial model training, the CNN was heavily biased towards the "No_Pattern" class. This is a classic real-world dataset problem, as typical market days rarely form perfect W-Bottoms or Head & Shoulders. 

Solution: I addressed this severe data imbalance by calculating and applying custom class_weight adjustments during model training. By penalizing the model more for misclassifying minority patterns, it successfully regained its sensitivity and accuracy in detecting rare but highly profitable chart setups without overfitting to the majority class.

---

## How to Use (Local Setup)

If you want to run this scanner on your local machine, follow these steps:

1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)Sam-hana/AI-Stock-Pattern-Scanner.git
cd AI-Stock-Pattern-Scanner

2. Install dependencies
Ensure you have Python installed, then run the following command to install the required packages:
pip install tensorflow yfinance mplfinance pandas requests numpy

3. Generate the Daily Report
Run the main script to fetch today's data, scan for patterns, and generate the HTML report:
python generate_report.py

4. Retrain the Model (Optional)
If you want to modify the architecture or train the model with new dataset images:
python train_model.py
```bash
## Current Limitations & Next Steps

While the system successfully demonstrates an end-to-end automated scanning pipeline, the CNN model is currently trained on a relatively small dataset. As a result, the model's generalization capabilities and absolute accuracy in highly volatile or unseen market conditions still have room for improvement. 

To address this and further enhance the model's robustness, the immediate next steps include:
* Dataset Expansion: Scraping a larger and more diverse set of historical K-line charts across different market cycles.
* Advanced Data Augmentation: Implementing more aggressive image augmentation techniques to simulate various market noise levels and prevent overfitting.
