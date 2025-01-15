import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to ChromeDriver
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"

def selenium_login_and_download(username, password, access_date, download_dir):
    """
    Log in to DRF using Selenium, navigate to the target page, and download the correct PDF.
    """
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-popup-blocking")

    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Step 1: Log in to DRF
        driver.get("https://www.drf.com/login?retUrl=https://www.drf.com/")
        logging.info("Opened DRF login page.")
        
        # Wait for email and password fields to load
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "email"))).send_keys(username)
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "password"))).send_keys(password)

        logging.info("Please complete the CAPTCHA manually in the browser.")
        input("Press Enter after completing the CAPTCHA and logging in manually...")
        logging.info("Login successful.")

        # Step 2: Navigate to the "Handicapping & PPs" page
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.LINK_TEXT, "Handicapping & PPs"))).click()
        logging.info("Navigated to 'Handicapping & PPs'.")

        # Step 3: Navigate to "Daily Racing Program"
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'drp-program')]"))).click()
        logging.info("Navigated to 'Daily Racing Program'.")

        # Step 4: Select the date
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, f"//span[text()='{access_date.day}']"))
        ).click()
        logging.info(f"Selected date: {access_date.strftime('%Y-%m-%d')}.")

        # Step 5: Click the "Access" button
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Access']"))
        ).click()
        logging.info("Clicked 'Access' button.")
        
        # Step 6: Extract the PDF URL
        pdf_url = driver.current_url
        logging.info(f"Constructed URL: {pdf_url}")

        # Step 7: Download the PDF
        response = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
        pdf_path = os.path.join(download_dir, "DRFPPS.pdf")
        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/pdf':
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded and saved PDF: {pdf_path}")
            driver.quit()
            return pdf_path
        else:
            logging.error(f"Failed to download PDF: Status {response.status_code}, Content-Type {response.headers.get('Content-Type')}")
            driver.quit()
            return None
    except Exception as e:
        logging.error(f"Error during Selenium login and PDF download: {e}")
        driver.quit()
        return None

def prepare_data(file_path):
    """
    Prepare data for LSTM model.
    """
    df = pd.read_csv(file_path)

    if 'winner' not in df.columns:
        raise ValueError("'winner' column is required in the dataset.")
    
    target = df['winner']  # Binary classification target column
    features = df.drop('winner', axis=1)
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    X = [features_scaled[:i+1] for i in range(len(features_scaled) - 1)]
    y = target[1:].values

    max_length = max(len(seq) for seq in X)
    X_padded = np.array([np.pad(seq, ((max_length - len(seq), 0), (0, 0)), mode='constant') for seq in X])

    return np.array(X_padded), np.array(y)

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    username = "alwhaire@aol.com"
    password = "TacoTerror1"
    access_date = datetime.strptime("2024-12-14", "%Y-%m-%d")
    download_dir = "/Users/alexanderhaire/Documents/RacePrograms"

    # Step 1: Login and download the PDF
    pdf_path = selenium_login_and_download(username, password, access_date, download_dir)
    if not pdf_path:
        logging.error("PDF download failed. Exiting.")
        return

    # Step 2: Prepare the data and train the LSTM model
    try:
        csv_file_path = "/path/to/your/csv_file.csv"  # Replace with your actual CSV file path
        X, y = prepare_data(csv_file_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)

        history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        logging.info(f"Test Accuracy: {test_accuracy:.2f}")

        model.save("horse_race_predictor_drf.h5")
        logging.info("Model training complete and saved as 'horse_race_predictor_drf.h5'.")
    except Exception as e:
        logging.error(f"Error during LSTM model training: {e}")

if __name__ == "__main__":
    main()
