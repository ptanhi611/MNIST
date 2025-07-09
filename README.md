# 🧠 MNIST Digit Recognizer from Scratch

A fully functional handwritten digit recognition system built **from scratch using only NumPy**, without any external machine learning frameworks. This project showcases the full pipeline: data preprocessing, neural network implementation, training, and an **interactive Streamlit web app** for drawing digits and getting predictions in real-time.

---

## 🚀 Demo

<img src="https://i.imgur.com/PLbhW9j.gif" width="600"/>

Draw a digit using your trackpad or touchscreen and the model predicts it instantly.

---

## 📌 Features

- ✅ **Custom Neural Network** built only with `numpy`
- 🧠 Uses **pre-trained parameters** (weights + biases)
- 🎨 Interactive canvas to draw digits (0–9)
- 📊 Live predictions with confidence scores and bar chart
- 🌐 Fully integrated with **Streamlit** frontend

---

## 🧱 Architecture Overview

Your neural network architecture (example used):
```
Input (28x28 = 784) →
  Dense (128) + ReLU →
  Dense (64)  + ReLU →
  Dense (10)  + Softmax
```

All activations, weight updates, and forward passes implemented **from scratch** with `numpy`.

---

## 🗂 File Structure

For simplicity, all files are kept in a **single directory**:

```
mnist_digit_app/
│
├── app.py                      # Streamlit frontend app
├── canvas.py                   # Drawing canvas interface
├── layers.py                   # Layer & activation classes (ReLU, Softmax)
├── model_loader.py             # Loads stored weights & biases into model
├── predict_single.py           # Run single-digit inference from image
├── model_params_stored_mnist.pk1  # Pre-trained model parameters (weights + biases)
├── requirements.txt            # Required Python packages
└── README.md                   # 📘 You're here!
```

---

## 📦 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/mnist_digit_app.git
cd mnist_digit_app
```

### 2. Create Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🧪 How It Works

- The model is trained separately and saved using:
  ```python
  pickle.dump({"weights": [...], "biases": [...]}, open("model_params_stored_mnist.pk1", "wb"))
  ```

- The app loads these weights into a neural network defined in `layers.py`
- When a user draws a digit, the image is:
  - Converted to grayscale
  - Resized to 28×28
  - Normalized and reshaped to a (1, 784) vector
- The vector is passed through the network for inference
- Prediction + confidence is displayed in real time

---

## 💡 Technologies Used

| Type           | Tools/Libs                     |
|----------------|-------------------------------|
| Core ML Logic  | `numpy`, `pickle`              |
| Frontend       | `streamlit`, `streamlit-drawable-canvas` |
| Image Handling | `Pillow` (`PIL`)               |

---

## 🛠️ Author’s Note

This project was made **entirely from scratch** — no `scikit-learn`, no `TensorFlow`, no `PyTorch`. Just raw `NumPy` matrix ops and manual neural net logic. The purpose is to **deeply understand how neural networks work under the hood**.

The interactive app makes it usable, testable, and fun to play with!

---

## 📷 Screenshots

<img src="https://i.imgur.com/npX9v1E.png" width="500" />
<img src="https://i.imgur.com/jV6OpTu.png" width="500" />

---

## 📬 Contact / Suggestions

Feel free to raise issues or suggest improvements by opening an issue or PR.

---

## 🏁 Future Improvements

- [ ] Add option to upload digit image
- [ ] Visualize hidden layer activations
- [ ] Add training UI with live loss plot


---


