# NEURAL-STYLE-TRANSFER

*COMPANY*: CODTECH IT SOLUTION

*NAME*: ADITYA BHARAT GORATE

*INTERN ID*: CTIS3040

*DOMAIN*: AI

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

---

📌 Project Overview

This project implements Neural Style Transfer (NST) using TensorFlow and a pre-trained VGG19 model.
It blends the content of one image with the artistic style of another to generate a stylized output image.
Example: Convert a normal photo into a painting style like Van Gogh or Picasso.

---

🚀 Features
- Uses VGG19 (pretrained on ImageNet) for feature extraction
- Separates content and style representations
- Applies Gram Matrix for style computation
- Customizable:
  1. Style weight
  2. Content weight
- Epochs & steps
- Saves stylized output automatically

---

🛠️ Tech Stack
- Python 3.10+
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

---

📂 Project Structure

Neural-Style-Transfer/

│── neural_style_transfer.py

│── content2.jpg

│── style2.jpg

│── output/

│    └── styled.png      (automatically get created. you just need to create output folder)

---

⚙️ Installation

1️⃣ Clone the repository

git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer

2️⃣ Install dependencies

pip install tensorflow numpy matplotlib pillow

---

▶️ How to Run

python neural_style_transfer.py

Make sure these images exist in the root folder:

- content2.jpg → Content image

- style2.jpg → Style image

---

🧠 How It Works

1. Load content & style images
2. Resize and normalize images
3. Pass images through VGG19
4. Extract:
   - Content features
   - Style features
5. Compute Gram Matrix for style
6. Calculate losses:
   - Content Loss
   - Style Loss
7. Optimize generated image using Adam Optimizer
8. Save final stylized image

---

📊 Loss Function

Total Loss = Content Loss + Style Loss

- Content Weight: 1e4

- Style Weight: 1e-2

These can be tuned for different artistic effects.

---

🖼️ Output

The stylized image is saved in:

output/styled.png

It is also displayed using Matplotlib.


