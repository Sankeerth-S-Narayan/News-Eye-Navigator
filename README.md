# NewsEye-Navigator
# Continent Map News App

This application displays a world map with clickable continents and shows relevant news headlines for each selected continent. It features both eye-tracking and hand gesture controls for an interactive user experience.

## Features

- Interactive world map with clickable continents
- Real-time news headlines for each continent
- Dual control modes: eye-tracking and hand gestures
- Automatic switching between control modes based on hand detection
- Scrollable news feed

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- feedparser
- Pillow
- mediapipe
- pyautogui
- pynput

## Installation


1. Install the required dependencies:

pip install -r requirements.txt


## Usage

2. Run the main application:

python app.py



3. The application window will open, showing the world map and news feed.
4. Use eye-tracking or hand gestures to control the cursor and interact with the map.
5. Click on a continent to view its news headlines.
6. Scroll through the news feed using gestures or eye movements.
7. Press 'q' to quit the application.

## Controls

- Eye-tracking mode: Move your eyes to control the cursor. Blink to click.
- Hand gesture mode: 
- Move your index finger to control the cursor
- Bend Index finger, keep middle finger straight, and thumb apart from index finger for left click
- Bend Index finger, bend middle finger, and thumb apart from index finger for scroll up
- Closed all four fingers apart from thumb for scroll down

The app automatically switches between eye-tracking and hand gesture modes based on hand detection.

## Note

Ensure your webcam is properly connected and accessible for the control features to work correctly.


