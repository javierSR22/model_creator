## SCRIPT CON LAS FUNCIONES PARA OPTIMIZAR LOS DISTINTOS MODELOS ##

# ----------------- Librerías ---------------------- #
from keras.utils import to_categorical                          # Codigicador One-hot de variables categóricas para modelos de Keras (esencial para clasificación multicategórica)
import keras_tuner as kt                                        # Librería para hacer hyperparameter tunning
from keras_tuner.oracles import BayesianOptimizationOracle            # Oracle de la optimización bayesiana (solo para modelos sklearn)
from keras_tuner.tuners import SklearnTuner                     # Tuner para modelos de Sklearn

# importamos "piezas" de la red neuronal
from keras.models import Sequential                             # Estructura básica de keras para crear redes neuronales
from keras.layers import Dense,Dropout                          # Capas densas y de drop-out para redes neuronales
from keras.optimizers import Adam,SGD                           # Optimizadores Adam y SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint       # Callbacks Earlystopping y Modelcheckpoint para ahorrar coste computacional y guardar el mejor modelo respectivamente

# importamos los modelos de Sklearn
from sklearn import svm                                        # Support Vector Machine
from sklearn.ensemble import RandomForestClassifier            # Random Forest Classifier
from sklearn.neighbors import KNeighborsClassifier             # K-Nearest Neighbors Classifier

# importamos las herramientas para obtener las métricas que guiarán la optimización bayesiana
from sklearn import metrics                                              # Métricas de Sklearn
from sklearn.model_selection import StratifiedKFold,train_test_split     # SKF-CV y Train Test Split para validar y entrenar modelos
from sklearn.preprocessing import LabelEncoder                           # Codificador de variables categóricas para modelos de Sklearn
# ----------------- Librerías ---------------------- #

# ---------------------------- Funciones de optimización de modelos ---------------------------- #

ov = True   # True: Sobrescribe los modelos actualmente guardados || False: No sobrescribe los modelos actuales, extrae resultados guardados (ideal para hacer pruebas de otras funciones generando menor gasto computacional)
metric = metrics.make_scorer(metrics.recall_score,needs_proba=False,average='macro')    # Métrica para los modelos de Sklearn

##################################
##--- Support vector Machine ---##

def SVM_Optimizer(X,y,weight=None):                                                             # Función para optimizar el modelo
    def build(hp):                                                                  # Definimos la función que construye el hipermodelo
        support = svm.SVC(                                                          # Creamos una variable que contenga al modelo SVM classifier
            C=hp.Int('C',1,10),                                                     # Definimos el rango de búsqueda para los valores del parámetro C
            kernel=hp.Choice('Kernel',values=['linear','poly','rbf','sigmoid']),    # Definimos las distintas opciones para el kernel
            probability=True,
            class_weight=weight
        )
        
        return support                                                              # Nos devuelve la configuración para la optimización de nuestro hipermodelo

    oracle = BayesianOptimizationOracle(                                            # Definimos los parámetros para la optimización bayesiana
        objective=kt.Objective('score',direction='max'),                            # Le decimos que la métrica que sea la tiene que maximizar
        max_trials=65                                                               # Se harán 65 combinaciones y se seleccionará la mejor
    )

    tuner = SklearnTuner(                                           # definimos el tuner que guiará la búsqueda
        oracle=oracle,                                              # Usamos oracle de la optimización bayesiana para decirle que queremos hacer ese tipo de búsqueda
        hypermodel=build,                                           # Especificamos qué hipermodelo queremos
        scoring=metric,                                             # Definimos que la puntuación vendrá dada por la AUC
        cv=StratifiedKFold(6),                                      # Definimos el tipo de Cross Validation como Stratified K Fold con 6 folds
        directory='.\\MachineLearning\\model_trials',               # Se guardarán los resultados en el directorio en el que se ejecute la función
        project_name='SVM_tuner',                                   # Definimos el nombre del proyecto bajo el que se guardan los resultados
        overwrite=ov                                                # Sobrescribimos los resultados cada vez que se ejecute la función
    )

    enc = LabelEncoder()    # Creamos un codificador de variables factor
    enc.fit(y)              # Lo entrenamos para que nombre apropiadamente nuestras variables [¡] Hay que cerciorarse de que los nombres son correctos [¡]
    y = enc.transform(y)    # Transformamos las categorías correspondientes a cada muestra

    X=X.to_numpy()          #transformamos la variable X a numpy array para entrenar el modelo en las mismas condiciones que se usarán despues

    tuner.search(X,y)       # Alimentado por los datos que metemos en la función, el optimizador busca los mejores parámetros

    tuner.results_summary() # Cuando termina la búsqueda te enseña un resumen de la misma

    best_model = tuner.get_best_hyperparameters()[0]    # Obtenemos los hiperparámetros del modelo que ha quedado en primer lugar

    # IMPRIMIMOS POR LA CONSOLA LOS MEJORES HIPERPARÁMETROS
    print('------------------ MEJORES PARÁMETROS PARA SVM ------------------')
    print(best_model.values)
    print('-----------------------------------------------------------------')

    modelo_seleccionado = tuner.hypermodel.build(best_model)        # Creamos el moejor modelo con los mejores hiperparámetros
    return modelo_seleccionado                                      # Devolvemos dicho modelo

##--- Support vector Machine ---##
##################################
##--- Random Forest ---##

def RF_Optimizer(X,y,weight=None):                                                                  # Función para optimizar el modelo
    def build(hp):                                                                      # Definimos la función que construye el hipermodelo
        support = RandomForestClassifier(                                               # Creamos una variable que contenga al modelo SVM classifier
            n_estimators=hp.Int('n_trees',10,500),                                      # Intervalo numérico para el número de árboles de decisión
            criterion=hp.Choice('Criterion',values = ['gini','entropy','log_loss']),     # Opciones para la función que mide la calidad de los splits
            class_weight=weight
        )
        
        return support                                                              # Nos devuelve la configuración para la optimización de nuestro hipermodelo

    oracle = BayesianOptimizationOracle(                                            # Definimos los parámetros para la optimización bayesiana
        objective=kt.Objective('score',direction='max'),                            # Le decimos que la métrica que sea la tiene que maximizar
        max_trials=65                                                               # Se harán 65 combinaciones y se seleccionará la mejor
    )

    tuner = SklearnTuner(                                                # definimos el tuner que guiará la búsqueda
        oracle=oracle,                                              # Usamos oracle de la optimización bayesiana para decirle que queremos hacer ese tipo de búsqueda
        hypermodel=build,                                           # Especificamos qué hipermodelo queremos
        scoring=metric,                                             # Definimos que la puntuación vendrá dada por la AUC
        cv=StratifiedKFold(6),                                      # Definimos el tipo de Cross Validation como Stratified K Fold con 6 folds
        directory='.\\MachineLearning\\model_trials',               # Se guardarán los resultados en el directorio en el que se ejecute la función
        project_name='RF_tuner',                                    # Definimos el nombre del proyecto bajo el que se guardan los resultados
        overwrite=ov                                              # Sobrescribimos los resultados cada vez que se ejecute la función
    )

    enc = LabelEncoder()    # Creamos un codificador de variables factor
    enc.fit(y)              # Lo entrenamos para que nombre apropiadamente nuestras variables [¡] Hay que cerciorarse de que los nombres son correctos [¡]
    y = enc.transform(y)    # Transformamos las categorías correspondientes a cada muestra

    X=X.to_numpy()          #transformamos la variable X a numpy array para entrenar el modelo en las mismas condiciones que se usarán despues 
    
    tuner.search(X,y)       # Alimentado por los datos que metemos en la función, el optimizador busca los mejores parámetros

    tuner.results_summary() # Cuando termina la búsqueda te enseña un resumen de la misma

    best_model = tuner.get_best_hyperparameters()[0]    # Obtenemos los hiperparámetros del modelo que ha quedado en primer lugar

    # IMPRIMIMOS POR LA CONSOLA LOS MEJORES HIPERPARÁMETROS
    print('------------------ MEJORES PARÁMETROS PARA RF ------------------')
    print(best_model.values)
    print('-----------------------------------------------------------------')

    modelo_seleccionado = tuner.hypermodel.build(best_model)
    return modelo_seleccionado

##--- Random Forest ---##
##################################
##--- K-Nearest Neighbors ---#

def KNN_Optimizer(X,y):                                                                 # Función para optimizar el modelo
    def build(hp):                                                                      # Definimos la función que construye el hipermodelo
        support = KNeighborsClassifier(                                                 # Creamos una variable que contenga al modelo SVM classifier
            n_neighbors=hp.Int('n_neighbors',5,int(X.shape[0]*0.75)),                            # Intervalo para decidir el número de vecinos a consultar con un máximo del 75% de observaciones en los datos 
            algorithm=hp.Choice('Algorithm',values=['auto','ball_tree','brute','kd_tree'])       # Opciones para el algoritmo que computa los vecinos más cercanos
        )
        
        return support                                                              # Nos devuelve la configuración para la optimización de nuestro hipermodelo

    oracle = BayesianOptimizationOracle(                                            # Definimos los parámetros para la optimización bayesiana
        objective=kt.Objective('score',direction='max'),                            # Le decimos que la métrica que sea la tiene que maximizar
        max_trials=65                                                               # Se harán 65 combinaciones y se seleccionará la mejor
    )

    tuner = SklearnTuner(                                           # definimos el tuner que guiará la búsqueda
        oracle=oracle,                                              # Usamos oracle de la optimización bayesiana para decirle que queremos hacer ese tipo de búsqueda
        hypermodel=build,                                           # Especificamos qué hipermodelo queremos
        scoring=metric,                                             # Definimos que la puntuación vendrá dada por la AUC
        cv=StratifiedKFold(6),                                      # Definimos el tipo de Cross Validation como Stratified K Fold con 6 folds
        directory='.\\MachineLearning\\model_trials',               # Se guardarán los resultados en el directorio en el que se ejecute la función
        project_name='KNN_tuner',                                   # Definimos el nombre del proyecto bajo el que se guardan los resultados
        overwrite=ov                                                # Sobrescribimos los resultados cada vez que se ejecute la función
    )

    enc = LabelEncoder()    # Creamos un codificador de variables factor
    enc.fit(y)              # Lo entrenamos para que nombre apropiadamente nuestras variables [¡] Hay que cerciorarse de que los nombres son correctos [¡]
    y = enc.transform(y)    # Transformamos las categorías correspondientes a cada muestra

    X=X.to_numpy()          #transformamos la variable X a numpy array para entrenar el modelo en las mismas condiciones que se usarán despues

    tuner.search(X,y)       # Alimentado por los datos que metemos en la función, el optimizador busca los mejores parámetros

    tuner.results_summary() # Cuando termina la búsqueda te enseña un resumen de la misma

    best_model = tuner.get_best_hyperparameters()[0]    # Obtenemos los hiperparámetros del modelo que ha quedado en primer lugar

    # IMPRIMIMOS POR LA CONSOLA LOS MEJORES HIPERPARÁMETROS
    print('------------------ MEJORES PARÁMETROS PARA KNN ------------------')
    print(best_model.values)
    print('-----------------------------------------------------------------')

    modelo_seleccionado = tuner.hypermodel.build(best_model)        # Construimos el mejor  modelo con los mejores hiperparámetros
    return modelo_seleccionado                                      # devolvemos dicho modelo

##--- K-Nearest Neighbors ---#
##################################
##--- Neural Network ---#

def MLP_Optimizer(X,y):                                             # Función para optimizar el modelo
    def build(hp):                                                  # Definimos la función que construye el hipermodelo
        mlp = Sequential()                                          # creamos la variable que contendrá al MLP
            
        activation = hp.Choice('Activation',['relu','elu','selu','tanh'], default = 'relu') # definimos las distintas funciones de activación posibles
        
        mlp.add(Dense(units = hp.Int('Dense',min_value=15,max_value=X.shape[1]*2), activation = activation,input_shape=(X.shape[1],)))   #definimos capa de input definieindo a su vez el rango de búsqueda para la capa densa de neuronas
        mlp.add(Dropout(hp.Choice('Dropout_init',values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))     # Definimos capa de droput para evitar sobreentrenamiento de la red

        # bucle para seleccionar el número de capas
        for i in range(hp.Int('num_dense_layers',2,6)):
            mlp.add(Dense(units=hp.Int('dense_' + str(i), min_value=15, max_value=(X.shape[1]*2)),activation=activation)) # Definimos capa de neuronas
            mlp.add(Dropout(hp.Choice('Dropout_'+ str(i),values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))     # Definimos capa de droput para evitar sobreentrenamiento de la red

        activation_out = hp.Choice('Act_out',['sigmoid','softmax'],default='sigmoid')       # Definimos las funciones de salida para la última capa
        mlp.add(Dense(4,activation=activation_out))                                         # Capa de salidad, 1 neurona x clase
        
        ## Defnimos los dos posibles optimizadores para la red neuronal
        optimizador = hp.Choice('Optimizer',values=['Adam','SGD'])  # Opciones de optimizador: ADAM y SGD
        l_rate = hp.Choice('Learning_rate', values=[0.01,0.001])    # Opciones de Learning Rate: 0.01 y 0.001   [¡] El valor 0.1 es demasiado rápido para datos de esta dimensionalidad por lo que no se incluye

        if optimizador == 'Adam':                               # Si el optimizador es Adam:
            opti = Adam(learning_rate=l_rate,)                  # Se establece Adam con la opción correspondiente de Learning Rate
        elif optimizador == 'SGD':
            momentum = hp.Choice('Momentum', values = [0.6, 0.7, 0.8, 0.9]) # Opciones para el valor de Momentum para SGD
            opti = SGD(learning_rate=l_rate,momentum=momentum)              # Se estabñece SGD con las opciones correspondientes de Learning Rate y Momentum

        # Compilamos nuestro modelo suponiendo que haremos clasificación multiclase y que nos muestre las métricas Accuracy y AUC durante el entrenamiento (para extraerlas después si queremos)
        mlp.compile(optimizer=opti, loss="categorical_crossentropy", metrics=['Accuracy','AUC'])

        return mlp                                                              # Nos devuelve la configuración para la optimización de nuestro hipermodelo


    obj = kt.Objective('val_Accuracy',direction='max')      # Definimos como objetivo maximizar la precisión de validación de nuestro modelo

    # Definimos el tuneador para nuestra optimización bayesiana de Keras
    tuner = kt.BayesianOptimization(
        build,                                          # Introducimos el hipermodelo
        seed=2233,                                      # Semilla para que nos proporcione siempre resultados iguales o en la misma línea
        objective=obj,                                  # Introducimos el objetivo definido anteriormente
        max_trials=100,                                 # Se realizarán 100 intentos como máximo para encontrar el mejor modelo
        directory='.\\MachineLearning\\model_trials',   # Se guarda la información sobre cada intento en el siguiente directorio
        project_name='tuning-mlp',                      # Se guarda el proyecto bajo el siguiente nombre
        overwrite=ov                                    # "botón" de overwrite para todas las optimizaciones bayesianas
    )

    #establecemos callbacks para reducir coste computacional
    es = EarlyStopping(monitor='val_Accuracy', mode='max', patience=100)                    # parará tras 100 epochs si la accuracy de validación no aumenta
    mc = ModelCheckpoint('.\\MachineLearning\\model_trials\\best_model_mlp.h5', monitor='val_Accuracy', mode='max', save_best_only=True)    # Guardará el último modelo que haya dado los mejores valores de Accuracy de validación

    X=X.to_numpy()   # transformamos la variable X a numpy array para entrenar el modelo en las mismas condiciones que se usarán despues

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, shuffle=True, random_state=2233)  # dividimos set de validación para red neuronal
    
    # Transformamos las categorías correspondientes a cada muestra
    y_train = to_categorical(y_train,num_classes=4)   
    y_test = to_categorical(y_test,num_classes=4)


    tuner.search(X_train,y_train,                   # Introducimos los datos de X e Y para entrenar el modelo en cada epoch
                 epochs=3000,                       # Establecemos que se llevarán a cabo 3000 epoch como máximo
                 validation_data=(X_test,y_test),   # Establecemos los siguientes datos de validación
                 callbacks=[es,mc],                 # Introducimos las callbacks configuradas anteriormente
                 verbose=1)                         # Pedimos que se nos muestre por pantalla el seguimiento de la optimización para cerciorarnos de que funciona y ver su evolución

    tuner.results_summary() # Cuando termina la búsqueda te enseña un resumen de la misma

    best_model = tuner.get_best_hyperparameters()[0]    # Obtenemos los hiperparámetros del modelo que ha quedado en primer lugar

    # IMPRIMIMOS POR LA CONSOLA LOS MEJORES HIPERPARÁMETROS
    print('------------------ MEJORES PARÁMETROS PARA MLP ------------------')
    print(best_model.values)
    print('-----------------------------------------------------------------')

    modelo_seleccionado = tuner.hypermodel.build(best_model)    # Construimos de nuevo el mejor modelo

    return modelo_seleccionado  # DEvuelve el mejor modelo seleccionado

##--- Neural Network ---#
# ---------------------------- Funciones de optimización de modelos ---------------------------- #