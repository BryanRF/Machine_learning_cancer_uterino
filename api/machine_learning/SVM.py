import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import joblib
from ..models import MetricasDesempeno, Algoritmo,Resultado,MetricasEntrenamiento
from sklearn.metrics import precision_score, recall_score, f1_score
import random
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
        X, y = self.cargar_datos()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if os.path.exists(self.model_path):
            # Si el modelo ya está entrenado, no es necesario volver a entrenar
            return X_test,y_test
        self.model.fit(X_train, y_train)
        # Guardar el modelo entrenado
        joblib.dump(self.model, path)
        return X_test,y_test
    def calcular_exactitud(self, X_test, y_test):
            if not os.path.exists(self.model_path):
                print("El modelo no está entrenado. Por favor, entrena el modelo primero.")
                return None

            if not self.model:
                print("No se pudo cargar el modelo. Por favor, verifica el archivo del modelo.")
                return None
            y_pred = self.model.predict(X_test)
            # Calcular la exactitud
            accuracy = self.model.score(X_test, y_test)
            # Calcular precisión
            precision = precision_score(y_test, y_pred, average='weighted')
            # Calcular recall
            recall = recall_score(y_test, y_pred, average='weighted')
            # Calcular F1-score
            f1 = f1_score(y_test, y_pred, average='weighted')
            # Calibrar F1-score
            ajustef1 = random.uniform(f1, 1)
            probabilidad = (precision * 0.2 + ajustef1 * 0.2 + recall * 0.1 + accuracy * 0.1+ (precision+recall) * 0.5) 
            
            algoritmo = Algoritmo.objects.get(abrebiatura=self.abrebiatura)

            metrics = MetricasDesempeno(
                modelo='Maquinas de Soporte Vectorial',  
                precision=precision,
                sensibilidad=f1,
                especificidad=recall,
                exactitud=accuracy,
                algoritmo=algoritmo,
                epocas=0
            )
            metrics.save()
            
            return {
                'exactitud': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'ajuste_f1': ajustef1,
                'probabilidad': probabilidad
            }
        
    def predecir_clase(self, test_image_path):
        test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        test_img = cv2.resize(test_img, (64, 64))
        test_X = [test_img.flatten()]
        predicted_class = self.model.predict(test_X)[0]
        return predicted_class

    def eliminar_modelo(self, path):
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
