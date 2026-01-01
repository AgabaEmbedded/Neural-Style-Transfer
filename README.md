# Neural Style Transfer Web App

<p align="center">
  <a href="https://neural-style-transfer-mfeke2scdkwbajkjsgwc4q.streamlit.app/">
    <img src="Demo.png" width="100" />
  </a>
</p>

A beautiful and interactive **Streamlit** web application that applies **Arbitrary Neural Style Transfer** to any uploaded image using the pre-trained Magenta model from TensorFlow Hub.

Transform your photos into stunning artworks inspired by 9 different artistic styles in seconds — powered entirely by deep learning!

## Live Demo
Run it locally and experience the magic!

## Features

- Upload any image (PNG, JPG, JPEG)
- Choose from **9 unique pre-defined artistic styles**
- Real-time style transfer using Google's **Magenta Arbitrary Image Stylization** model
- Side-by-side comparison: Original → Style → Stylized Result
- High-resolution downloadable output (1024×1024 PNG)
- Clean, responsive, and user-friendly interface
- Fast inference with cached model and image loading

## Screenshots

*(Add your own screenshots later by replacing this section)*

## Repository Structure

```
Neural-Style-Transfer/
├── nst.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── Style 1.jpg                # Pre-defined style images (1 to 9)
├── Style 2.jpg
├── ...
└── Style 9.jpg
```

## Requirements

- Python 3.8+
- Streamlit
- TensorFlow (2.x)
- TensorFlow Hub
- Pillow (PIL)
- NumPy

Install all dependencies easily with:

```bash
pip install -r requirements.txt
```

### requirements.txt content (for reference)
```
streamlit
tensorflow
tensorflow-hub
Pillow
numpy
```

## How to Run

1. Clone or download this repository
2. Navigate to the project folder:
   ```bash
   cd Neural-Style-Transfer
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run nst.py
   ```
5. Open your browser (usually at `http://localhost:8501`)
6. Upload an image, select a style, and watch the AI create art!

## How It Works

- Uses the official **[Magenta Arbitrary Image Stylization v1-256](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)** model from TensorFlow Hub
- Automatically crops input to square, resizes to 256×256 for optimal processing
- Applies fast style transfer combining content and selected style image
- Upscales final result to 1024×1024 for high-quality downloads

## Credits

- Model: [Magenta Team @ Google](https://magenta.tensorflow.org/)
- Neural Style Transfer Research: Gatys et al., and fast arbitrary stylization by Google
- Built with ❤️ using Streamlit and TensorFlow

## Author

**Agaba_Embedded**  
Transforming pixels into art, one upload at a time.

---

**Enjoy creating AI art!** 🖼✨  
Feel free to fork, star ⭐, and customize with your own styles!
