## SCRIPT CON LAS FUNCIONES PARA OBTENER MÉTRICAS MEDIANTE CROSS VALIDATION ##

# ----------------- Librerías ---------------------- #
from sklearn import metrics                                             # Importamos todas las métricas de Sklearn
from sklearn.model_selection import StratifiedKFold                     # Importamos SKF-CV para obtener las métricas
from keras.callbacks import EarlyStopping,ModelCheckpoint               # Importamos las callbacks Earlystopping y Modelcheckpoint para ahorrar coste computacional y guardar el mejor modelo respectivamente
import numpy as np                                                      # Numpy es necesaria para introducir los datos
from sklearn.preprocessing import LabelEncoder                          # Codificador de variables categóricas para modelos Sklearn
from keras.utils import to_categorical                                  # Codigicador One-hot de variables categóricas para modelos de Keras (esencial para clasificación multicategórica)
# ----------------- Librerías ---------------------- #

# NOTA: Es preferible usar LOO-CV por ser más exacta a la hora de obtener las métricas, pero para 120 muestras puede ser computacionalmente
#       demasiado costoso para la red neuronal. Por este motivo, voy a usar SKF agrupando las muestras según su nivel de Diesel para mantener la proporción de los grupos
#       en todo momento.


# --------------------- Funciones de Cross Validation --------------------- #

# función de CV para modelos de Sklearn

def Sklearn_CV(X,y,model):
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state= 12)    # Creamos Stratified K fold con 6 splits y barajando los datos

    enc = LabelEncoder()    # Creamos un codificador de variables factor
    enc.fit(y)              # Lo entrenamos para que nombre apropiadamente nuestras variables [¡] Hay que cerciorarse de que los nombres son correctos [¡]
    y = enc.transform(y)    # Transformamos las categorías correspondientes a cada muestra

    X=X.to_numpy()          #transformamos la variable X a numpy array para entrenar el modelo en las mismas condiciones que se usarán despues

    acc_per_fold = []                                                   # Vector vacío para guardar la accuracy
    ROC_auc_per_fold = []                                               # Vector vacío para guardar la AUC
    n_fold = 1                                                          # Contador para el número de Folds

    for train_index,test_index in cv.split(X,y):                        # Definimos el bucle para la Cross Validation
        X_train, X_test = X[train_index], X[test_index]                 # En cada loop se dividen los datos de training y test
        y_train, y_test = y[train_index], y[test_index]                 # En cada loop se dividen las categorías para training y test
        
        print('------------------------------------------------------------------------')
        print(f'Entrenamiento del modelo - fold {n_fold}')              # Imprime el número del fold por la pantalla
        model.fit(X_train,y_train)                                      # Entrena el primer modelo
        print('Finalizado')                                             # Imprime cuando finaliza el entrenamiento

        pred = model.predict(X_test)                                    # Predice resultados
        model_proba = model.predict_proba(X_test)                       # Predice las probabilidades de cada clase

        # Generamos las métricas de accuracy
        acc=metrics.accuracy_score(y_true=y_test,y_pred=pred)           # Extrae la métrica de precisión
        print('metrica Accuracy extraida')

        # Generámos las métricas de AUC para cada clase
        roc_auc = metrics.roc_auc_score(y_true=y_test,y_score=model_proba,multi_class='ovr')        # Calculamos roc auc score con las probabilidades de cada clase mediante One vs Rest
        print('métrica ROC AUC Score extraida')

        print(f'Métrica para el fold {n_fold}: Accuracy of {acc}; ROC AUC Score of {roc_auc}')

        acc_per_fold.append(acc)                # guardamos accuracy en el vector de accuracies
        ROC_auc_per_fold.append(roc_auc)        # guardamos auc en el vector de aucs

        # Incrementamos el contador de Folds en 1
        n_fold = n_fold + 1
    
    # Obtenemos métricas medias
    print('------------------------------------------------------------------------')
    print('Métricas por fold:')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Accuracy: {acc_per_fold[i]} - ROC AUC Score: {ROC_auc_per_fold[i]}')
    print('------------------------------------------------------------------------')
    print('Media de las métricas de todos los Folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> AUC Score: {np.mean(ROC_auc_per_fold)} (+- {np.std(ROC_auc_per_fold)})')
    print('------------------------------------------------------------------------')

    acc=np.mean(acc_per_fold)
    auc=np.mean(ROC_auc_per_fold)

    return acc,auc

# función de CV para modelos de Keras

def Keras_CV(X,y,model):

    cv=StratifiedKFold(n_splits=6,shuffle=True,random_state=12)         # Configuramos la validación cruzada

    X=X.to_numpy()   # transformamos la variable X a numpy array para entrenar el modelo en las mismas condiciones que se usarán despues
    

    acc_per_fold=[]                                                     # creamos un vector vacio para almacenar las métricas accuracy
    loss_per_fold=[]                                                    # creamos un vector vacio para almacenar las métricas de loss
    auc_per_fold=[]                                                     # creamos un vector vacio para almacenar las métricas de auc

    n_fold = 1                                                          # Empezamos por el fold 1

    for train_index, test_index in cv.split(X,y):                       # Bucle de Cross Validation
        X_train, X_test = X[train_index], X[test_index]                 # Separamos train y test para X
        y_train, y_test = y[train_index], y[test_index]                 # Separamos train y test para Y

        y_train = to_categorical(y_train,num_classes=4)                 # Convertimos las respuestas de entrenamiento en categorías de Keras (importante para que funcione el algoritmo)
        y_test = to_categorical(y_test,num_classes=4)                   # Convertimos las respuestas de test en categorías de Keras (importante para que funcione el algoritmo)

        es = EarlyStopping(monitor='Accuracy', mode='max', patience=100)        # Configuramos un early stopping para que se pare de entrenar cuando la accuracy no aumente tras 100 intentos
        mc = ModelCheckpoint('.\\MachineLearning\\model_trials\\best_model_mlpCV.h5', monitor='loss', mode='min', save_best_only=True)      # Configuramos que guarde el modelo que minimice la pérdida (lo mismo que maximizar auc)

        print('------------------------------------------------------------------------')
        print(f'Training para Neural Network - fold {n_fold}')
        model.fit(X_train,y_train,epochs=3000,callbacks=[es,mc],verbose=1)                  # Entrenamiento de la neural network para el fold n_fold. Se entrenarán 3000 epocas y se usaran las callbacks. [¡] Para que no se imprima todo el proceso pon verbose = 0
        print('Finalizado')


        # Generate evaluation metrics
        scores = model.evaluate(X_test, y_test, verbose=0)                                  # Se optienen las métricas del modelo usando la función evaluate del propio modelo
        print(f'Métricas del fold {n_fold}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}; {model.metrics_names[2]} of {scores[2]}') # Imprimimos las métricas del fold correspondiente
        
        loss_per_fold.append(scores[0])     # Añadimos loss a su vector correspondiente
        acc_per_fold.append(scores[1])      # Añadimos accuracy a su vector correspondiente
        auc_per_fold.append(scores[2])      # Añadimos AUC a su vector correspondiente

        # Aumentamos el número de fold
        n_fold = n_fold + 1
    
    # Imprimimos los resultados de todos los folds por separado y la media de los resultados
    print('------------------------------------------------------------------------')
    print('Métricas por fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]} - AUC: {auc_per_fold[i]}')
    print('------------------------------------------------------------------------')
    print('Métricas medias entre todos los folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> AUC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    acc=np.mean(acc_per_fold)   # calculamos la media de Accuracy
    auc=np.mean(auc_per_fold)   # calculamos la media de Auc

    return acc,auc      # Devolvemos la accuray y la AUC

# --------------------- Funciones de Cross Validation --------------------- #