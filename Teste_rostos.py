import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

def capture_images(person_name, num_images=20, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    save_path = os.path.join('dataset', person_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Capturing Images', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('0'):
            break
        elif key & 0xFF == ord(' '):  
            cv2.imwrite(os.path.join(save_path, f'image{count}.jpg'), frame)
            count += 1
        
    cap.release()
    cv2.destroyAllWindows()

def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if os.path.isdir(person_folder):
            label_dict[current_label] = person_name
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(current_label)
            current_label += 1

    return np.array(images), np.array(labels), label_dict

def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    return img

def train_model(images, labels, label_dict):
    labels = to_categorical(labels, num_classes=len(label_dict))

    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(label_dict), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=10, validation_split=0.2)
    
    model.save('face_recognition_model.keras')
    np.save('label_dict.npy', label_dict)
    return model

def load_trained_model():
    model = tf.keras.models.load_model('face_recognition_model.keras')
    label_dict = np.load('label_dict.npy', allow_pickle=True).item()
    return model, label_dict

def recognize_faces(model, label_dict, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            img = preprocess_image(face)
            prediction = model.predict(img)
            predicted_label = np.argmax(prediction)
            confidence = prediction[0][predicted_label]

            if confidence > 0.8:  
                name = label_dict[predicted_label]
            else:
                name = 'Unknown'

            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    camera_index = int(input("Digite o índice da câmera a ser usada (geralmente 0 ou 1): "))
    
    while True:
        print("\nMenu:")
        print("1. Cadastrar Pessoa")
        print("2. Editar Nome de Pessoa")
        print("3. Adicionar Imagens a uma Pessoa")
        print("4. Treinar Modelo")
        print("5. Reconhecimento em Tempo Real")
        print("6. Sair")
        
        choice = input("Escolha uma opção: ")

        if choice == '1':
            person_name = input("Digite o nome da pessoa: ")
            capture_images(person_name, camera_index=camera_index)
        elif choice == '2':
            old_name = input("Digite o nome atual da pessoa: ")
            new_name = input("Digite o novo nome da pessoa: ")
            os.rename(os.path.join('dataset', old_name), os.path.join('dataset', new_name))
        elif choice == '3':
            person_name = input("Digite o nome da pessoa: ")
            capture_images(person_name, camera_index=camera_index)
        elif choice == '4':
            images, labels, label_dict = load_images_from_folder('dataset')
            model = train_model(images, labels, label_dict)
        elif choice == '5':
            model, label_dict = load_trained_model()
            recognize_faces(model, label_dict, camera_index=camera_index)
        elif choice == '6':
            break
        else:
            print("Opção inválida. Por favor, tente novamente.")

if __name__ == "__main__":
    main_menu()
