import os
import pickle
import numpy as np
import librosa
import sounddevice as sd
import csv
from flask import Flask, render_template, jsonify
from datetime import datetime
from threading import Timer

app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\HomePC\Desktop\RainfallDetector\app\model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Predictions history and log path
predictions_history = []
log_path = r'C:\Users\HomePC\Desktop\RainfallDetector\logs'


def record_audio(duration=60, samplerate=44100):
    """Record audio from microphone for the given duration and samplerate."""
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='float32')
    sd.wait()  # Wait until recording is done
    print("Recording complete")
    return audio_data


def extract_features(audio_data, samplerate=44100):
    """Extract MFCC features from audio data."""
    audio = audio_data[:, 0]  # Use only one channel of the stereo signal (first channel)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=40).T, axis=0)
    return mfccs


def predict_intensity(audio_data, samplerate=44100):
    """Predict the rain intensity using the trained model."""
    features = extract_features(audio_data, samplerate)
    prediction = model.predict([features])
    return prediction[0]  # Return the predicted label (e.g., High, Moderate, Low, or No Rain)


def generate_flood_risk():
    """Generate flood risk based on the intensity readings in the last 60 minutes."""
    light_rain_count = sum(1 for p in predictions_history if p['intensity'] == 'Light Rain')
    moderate_rain_count = sum(1 for p in predictions_history if p['intensity'] == 'Moderate Rain')
    heavy_rain_count = sum(1 for p in predictions_history if p['intensity'] == 'Heavy Rain')
    no_rain_count = sum(1 for p in predictions_history if p['intensity'] == 'No Rain')

    if no_rain_count >= 60:
        return 'No Risk'

    if light_rain_count > 288:
        return 'Moderate Risk'
    elif light_rain_count >= 144:
        return 'Low Risk'
    elif moderate_rain_count >= 144:
        return 'Moderate Risk'
    elif heavy_rain_count > 36:
        return 'High Risk'
    elif heavy_rain_count >= 12:
        return 'Moderate Risk'
    else:
        return 'No Risk'


def record_and_predict():
    """Record audio, predict intensity, and log the result every 5 minutes."""
    audio_data = record_audio(duration=5)
    intensity = predict_intensity(audio_data)
    prediction_data = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'intensity': intensity,
        'flood_prediction': generate_flood_risk()
    }
    predictions_history.append(prediction_data)
    if len(predictions_history) > 20:
        predictions_history.pop(0)

    # Save to CSV file
    date_str = datetime.now().strftime('%Y-%m-%d')
    csv_file_path = os.path.join(log_path, f"{date_str}flood.csv")
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=prediction_data.keys())
        if f.tell() == 0:  # Write header only if file is empty
            writer.writeheader()
        writer.writerow(prediction_data)

    # Schedule the next prediction in 5 minutes
    Timer(300, record_and_predict).start()  # 300 seconds = 5 minutes


@app.route('/')
def home():
    # Start the recording and prediction process in the background if it's not already running
    if len(predictions_history) == 0:
        record_and_predict()

    # Display latest 20 predictions
    return render_template('rainweb.html', data=predictions_history[:20])


@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    """Endpoint to return the latest 20 predictions as JSON."""
    return jsonify(predictions_history[:20])


if __name__ == '__main__':
    app.run(debug=True)
