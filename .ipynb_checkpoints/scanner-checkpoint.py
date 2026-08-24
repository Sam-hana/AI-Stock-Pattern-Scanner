import os
import numpy as np
import tensorflow as tf
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import requests

def run_scan():
    # 1. Load the trained model
    print("Loading CNN model...")
    model = tf.keras.models.load_model('stock_pattern_cnn.keras')
    # Define classes ( match the alphabetical order of dataset folders)
    class_names = ['Head_Shoulder', 'No_Pattern', 'W_Bottom']
    
    print("Loading list of stocks...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        # Use request to access wikipedia and retrieve the data
        response = requests.get(url, headers=headers)
        response.raise_for_status() # throw exception if something happens
        
        
        tables = pd.read_html(response.text)
        
        # Get the list of stocks
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].tolist()
        
        #  change BRK.B to BRK-B
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"✅ Load {len(tickers)} stocks successfully！")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        
    # tables = pd.read_html(url)
    # sp500_df = tables[0]
    # tickers = sp500_df['Symbol'].tolist()
    
    # # The observing stock list
    # tickers = [ticker.replace('.', '-') for ticker in tickers] 
    
    # 3. Setup temporary folder for daily image generation
    temp_folder = "Temp_Scan"
    os.makedirs(temp_folder, exist_ok=True)
    
    # Re-use the exact same chart style from training to avoid confusing the model
    # For the model to use
    mc_model = mpf.make_marketcolors(up="g", down="r", edge="inherit", wick="inherit")
    style_model = mpf.make_mpf_style(marketcolors=mc_model, facecolor='black', edgecolor='black', figcolor='black', gridstyle="")
    
    # 2. For showing
    style_human = mpf.make_mpf_style(marketcolors=mc_model, gridstyle="--")
    
    print("Starting market scan...\n")
    print("-" * 50)
    detected_targets = []
    # 4. Automate the scanning process
    for ticker in tickers:
        try:
            # Fetch the latest 60 days of data
            df = yf.download(ticker, period="60d", interval="1d", multi_level_index=False)
            
            if len(df) < 60:
                continue
    
            # Define temporary file path
            filepath = os.path.join(temp_folder, f"{ticker}_latest.png")
            
            # Generate the K-line image (invisible to the user)
            mpf.plot(
                df, type="candle", style=style_model, axisoff=True,
                savefig=dict(fname=filepath, dpi=64, pad_inches=0)
            )
            
            # 5. Image Preprocessing for Prediction
            # Load image, resize to 64x64, and convert to array
            img = tf.keras.utils.load_img(filepath, target_size=(64, 64))
            img_array = tf.keras.utils.img_to_array(img)
            # Expand dimensions to match the batch format: (1, 64, 64, 3)
            img_array = tf.expand_dims(img_array, 0) 
    
            # 6. Make the Prediction
            predictions = model.predict(img_array, verbose=0)
            probabilities = predictions[0]
            
            predicted_index = np.argmax(probabilities)
            predicted_class = class_names[predicted_index]
            confidence = probabilities[predicted_index] * 100
            
            # 7. Output Alert filtering (Only alert if it's a specific pattern with decent confidence)
            if predicted_class != "No_Pattern" and confidence > 50.0:
                print("\n" + "🔥" * 25)
                print(f"🚨 TARGET IDENTIFIED: {ticker}")
                print(f"📈 Pattern: {predicted_class}")
                print(f"🎯 Confidence: {confidence:.2f}%")
                human_filepath = f"{temp_folder}/{ticker}_report.png"
                mpf.plot(
                    df, type="candle", style=style_human, title=f"{ticker} - Last 60 Days", volume=False,
                    savefig=dict(fname=human_filepath, dpi=100)
                )
                detected_targets.append({
                        "ticker": ticker,
                        "pattern": predicted_class,
                        "confidence": confidence,
                        "image_path": f"{temp_folder}/{ticker}_report.png"
                    })
                # Show the K line image to user 
                # mpf.plot(df, type="candle", style=style_human, title=f"{ticker} - Last 60 Days", volume=False)
                # print("🔥" * 25 + "\n")
            else:
                print(f"[{ticker}] Status normal. ")
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    print("-" * 50)
    print("\nScan completed!")
    return detected_targets
if __name__=="__main__":
    results=run_scan()
    print(f"Test ends, totally find{len(results)}")