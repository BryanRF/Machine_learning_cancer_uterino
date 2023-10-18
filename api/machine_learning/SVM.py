import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import joblib
class SVMClassifier:
    def __init__(self):
        self.abrebiatura = 'SVM'
        # self.model_path = Path("machine_learning", "entrenamiento", f"SVM.joblib")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.rd_name = f"SVM.joblib"
        self.model_path = os.path.join(current_dir,"entrenamiento",  self.rd_name)
        if os.path.exists(self.model_path):
            # Si el modelo ya existe, cargarlo
            self.model = joblib.load(self.model_path)
        else:
            # Si el modelo no existe, crear uno nuevo
            print(self.model_path)
            self.model = SVC(kernel='linear', probability=True)

    def cargar_datos(self):
        X = []
        y = []

        for clase in tqdm(os.listdir("media")):  # Agregamos tqdm aquí
            clase_path = os.path.join("media", clase)
            for imagen in os.listdir(clase_path):
                imagen_path = os.path.join(clase_path, imagen)
                img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))
                X.append(img.flatten())
                y.append(clase)

        return X, y

    def entrenar_modelo(self,path):
        if os.path.exists(self.model_path):
            # Si el modelo ya está entrenado, no es necesario volver a entrenar
            return

        X, y = self.cargar_datos()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for i in tqdm(range(100)):  
            self.model.fit(X_train, y_train)
        
        # Guardar el modelo entrenado
        joblib.dump(self.model, path)
        
    def predecir_clase(self, test_image_path):
        test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        test_img = cv2.resize(test_img, (64, 64))
        test_X = [test_img.flatten()]
        predicted_class = self.model.predict(test_X)[0]
        return predicted_class

    def guardar_imagen_prueba(self, imagen):
        with open('analisis/test_image.bmp', 'wb') as f:
            for chunk in imagen.chunks():
                f.write(chunk)
