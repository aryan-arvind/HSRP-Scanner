# Neural HSRP Intelligence: Real-Time Vehicle & License Plate Analytics

A deep learning-powered system designed for the automated detection, tracking, and recognition of High-Security Registration Plates (HSRP). This project leverages state-of-the-art computer vision models to provide a seamless video analytics experience.

## 🚀 Features
- **Real-Time Vehicle Tracking**: Utilizes YOLOv8 for robust vehicle detection and the SORT (Simple Online and Realtime Tracking) algorithm for persistent ID assignment.
- **HSRP Recognition**: Dedicated YOLOv8 model optimized for license plate localization.
- **Automated OCR**: Integrates EasyOCR with custom formatting logic to extract alphanumeric characters from HSRPs with high precision.
- **Interactive UI**: A Streamlit-based dashboard for easy video uploads, real-time visualization of results, and CSV data export.
- **Data Export**: Automatically generates detailed detection reports including timestamps, car IDs, and recognized plate numbers.

## 🛠️ Tech Stack
- **Deep Learning**: YOLOv8 (Ultralytics)
- **OCR Engine**: EasyOCR
- **Tracking**: SORT Algorithm
- **Web Interface**: Streamlit
- **Data Processing**: OpenCV, Pandas, NumPy

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Neural-HSRP-Intelligence.git
   cd Neural-HSRP-Intelligence
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run main.py
   ```
2. **Upload a video**: Use the dashboard to upload traffic footage.
3. **Process**: Click "Start Detection" to begin the analysis.
4. **Download**: Once complete, download the generated `detection_results.csv`.

## 📂 Project Structure
- `main.py`: The entry point for the Streamlit application.
- `util.py`: Helper functions for OCR, plate formatting, and CSV logging.
- `visualize.py`: Script for rendering detection overlays on video frames.
- `models/`: Contains the pre-trained YOLOv8 weights for license plate detection.
- `train/`, `valid/`, `test/`: Dataset partitions for model training.

## 📄 License
This project is licensed under the CC BY 4.0 License. See `data.yaml` for dataset licensing details.
