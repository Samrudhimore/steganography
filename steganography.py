import cv2
import os
import random
import hashlib
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simple neural network to generate a dynamic key
def create_key_generator_model():
    model = Sequential([
        Dense(32, input_shape=(32,), activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='linear')
    ])
    return model

def generate_dynamic_key(password, model):
    # Hash the password to create a fixed-length input
    password_hash = hashlib.sha256(password.encode()).digest()
    input_data = np.array(list(password_hash)).reshape(1, 32)
    
    # Use the model to generate a dynamic key
    dynamic_key = model.predict(input_data)[0]
    dynamic_key = dynamic_key.tobytes()[:16]  # Use the first 16 bytes as the key
    return dynamic_key

def encrypt_message(message, key):
    cipher = AES.new(key, AES.MODE_ECB)
    encrypted_message = cipher.encrypt(pad(message.encode(), AES.block_size))
    return encrypted_message

def decrypt_message(encrypted_message, key):
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted_message = unpad(cipher.decrypt(encrypted_message), AES.block_size)
    return decrypted_message.decode()

def embed_message(img, message):
    message_length = len(message)
    img_height, img_width, _ = img.shape

    # Embed the message length in the first 4 pixels
    for i in range(4):
        img[0, i, 0] = (message_length >> (8 * i)) & 0xFF

    # Embed the message in random pixels
    random.seed(42)  # Use a fixed seed for reproducibility
    for i in range(message_length):
        char = message[i]
        n = random.randint(0, img_height - 1)
        m = random.randint(0, img_width - 1)
        z = random.randint(0, 2)
        img[n, m, z] = ord(char)

    return img

def extract_message(img):
    # Extract the message length from the first 4 pixels
    message_length = 0
    for i in range(4):
        message_length |= img[0, i, 0] << (8 * i)

    # Extract the message from random pixels
    message = ""
    random.seed(42)  # Use the same seed as during embedding
    for i in range(message_length):
        n = random.randint(0, img.shape[0] - 1)
        m = random.randint(0, img.shape[1] - 1)
        z = random.randint(0, 2)
        message += chr(img[n, m, z])

    return message

def main():
    try:
        # Load the AI model for dynamic key generation
        key_generator_model = create_key_generator_model()

        img_path = input("Enter the image path: ").strip('"')  # Remove extra quotes
        if not os.path.exists(img_path):
            logging.error("Image file not found!")
            return

        img = cv2.imread(img_path)
        if img is None:
            logging.error("Failed to load the image!")
            return

        msg = input("Enter secret message: ")
        password = input("Enter a passcode: ")

        # Generate a dynamic key using the AI model
        dynamic_key = generate_dynamic_key(password, key_generator_model)

        # Encrypt the message
        encrypted_message = encrypt_message(msg, dynamic_key)

        # Embed the encrypted message into the image
        img = embed_message(img, encrypted_message.hex())

        # Save the encrypted image
        encrypted_img_path = "encryptedImage.jpg"
        cv2.imwrite(encrypted_img_path, img)
        logging.info(f"Encrypted image saved as {encrypted_img_path}")

        # Open the encrypted image
        os.system(f"start {encrypted_img_path}")  # Use 'start' to open the image on Windows

        # Decryption
        pas = input("Enter passcode for Decryption: ")
        if pas == password:
            # Generate the same dynamic key for decryption
            dynamic_key = generate_dynamic_key(pas, key_generator_model)

            # Extract the encrypted message from the image
            extracted_encrypted_message = extract_message(img)

            # Decrypt the message
            decrypted_message = decrypt_message(bytes.fromhex(extracted_encrypted_message), dynamic_key)
            logging.info(f"Decrypted message: {decrypted_message}")
        else:
            logging.error("YOU ARE NOT authorized!")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
