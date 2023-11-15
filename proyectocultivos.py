import tensorflow as tf
import numpy as np
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt


oculta1 = tf.keras.layers.Dense(units=3, input_shape=[3])
oculta2 = tf.keras.layers.Dense(units=3)
oculta3 = tf.keras.layers.Dense(units=3)
oculta4 = tf.keras.layers.Dense(units=3)
oculta5 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)


print("---------------------------------------------------------")
print("---- Bienvenido al Sistema de Prediccion de Cultivos ----")
print("---------------------------------------------------------")

salir = False
while salir == False:
    print("---------------------------------------------------------")
    print("1. Entrenar Software de Calidad de Caña ")
    print("2. Hacer Predicciones Calidad de Caña")
    print("3. Variables Internas del Modelo ")
    print("4. Salir ")
    print("---------------------------------------------------------")
    opc = input("Ingrese la Opcion: ")
    print("---------------------------------------------------------")

    if opc == '1':
        conexion = mysql.connector.connect(user='sql9640543', password='kChL6UUNWL', host='sql9.freesqldatabase.com', database='sql9640543', port='3306')

       #Recoleccion de Datos de la calidad de la caña
        mycursor = conexion.cursor()
        mycursor.execute("SELECT * FROM cultivos_entrenamiento")
        myresult = mycursor.fetchall()

        ndwi = []
        edad = []
        humedad = []
        Rendimiento = []

        # Procesar los resultados y guardar los datos en las listas
        for row in myresult:
            ndwi.append(row[2])      # La columna 2 corresponde a la NDWI
            edad.append(row[3])       # La columna 3 corresponde a las edad
            humedad.append(row[4])     # La columna 4 corresponde a la humedad
            Rendimiento.append(row[5])  # La columna 5 corresponde al rendimiento


        # Modelo de Red Neuronal para la Calidad de la caña
        ndwi = np.array(ndwi, dtype=float)
        edad = np.array(edad, dtype=float)
        humedad = np.array(humedad, dtype=float)
        Rendimiento = np.array(Rendimiento, dtype=float)

        oculta1 = tf.keras.layers.Dense(units=6, input_shape=[3])
        oculta2 = tf.keras.layers.Dense(units=6)
        oculta3 = tf.keras.layers.Dense(units=6)
        salida = tf.keras.layers.Dense(units=1)
        modelo = tf.keras.Sequential([oculta1, oculta2, oculta3, salida])

        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss='mean_squared_error'
        )

        X = np.column_stack((ndwi, edad, humedad))

        print("Comenzando entrenamiento...")
        # Se debe pasar X y Productividad como argumentos en la función fit()
        historial = modelo.fit(X, Rendimiento, epochs=10000, batch_size=1, verbose=False)
        print("Modelo entrenado!")

        #Entrenamiendo en base caña


    if opc == '2':

        conexion = mysql.connector.connect(user='sql9640543', password='kChL6UUNWL', host='sql9.freesqldatabase.com', database='sql9640543', port='3306')

        mypredict = conexion.cursor()
        mypredict.execute("SELECT * FROM cultivos_prediccion")
        predict = mypredict.fetchall()
        
        codigo = []
        nombre = []

        ndwip = []
        edadp = []
        humedadp = []

        rendimientop = []
        fechacortei = []
        fechacortef = []
        dias = []

        for row in predict:
            codigo.append(row[0])
            nombre.append(row[1])

            ndwip.append(row[2])      # La columna 2 corresponde a la NDWI
            edadp.append(row[3])       # La columna 3 corresponde a las edad
            humedadp.append(row[4])     # La columna 4 corresponde a la humedad

            fechacortei.append(row[6])
            fechacortef.append(row[7])
            dias.append(row[8])

        nueva_entrada = np.array([ndwip, edadp, humedadp], dtype=float)  # Crear un array con los datos
        nueva_entrada = nueva_entrada.T  # Transponer el array para que coincida con el formato esperado por el modelo
        resultado = modelo.predict(nueva_entrada)

        productic = pd.DataFrame(columns=['Codigo', 'Nombre Finca', 'NDWI', 'Edad', 'Humedad', 'Rendimiento', 'Fecha Corte Inicial', 'Fecha Corte Final', 'Dias'], index=range(len(resultado)))
        for i in range(len(resultado)):
            productic.iloc[i,0] = codigo[i]
            productic.iloc[i,1] = nombre[i]
            productic.iloc[i,2] = ndwip[i]
            productic.iloc[i,3] = edadp[i]
            productic.iloc[i,4] = humedadp[i]
            productic.iloc[i,5] = resultado[i][0]
            productic.iloc[i,6] = fechacortei[i]
            productic.iloc[i,7] = fechacortef[i]
            productic.iloc[i,8] = dias[i]

        productic = productic.sort_values(by='Rendimiento', ascending=False)
        print(productic)

        for i in range(len(resultado)):
            Sec = productic.iloc[i, 0]
            Pro = productic.iloc[i, 5]
            plt.bar(Sec, Pro)
        plt.show()
        plt.close('all')
        
    if opc == '3':

        print("Variables internas del modelo")
        #print(capa.get_weights())
        print(oculta1.get_weights())
        print(oculta2.get_weights())
        print(oculta3.get_weights())
        print(oculta4.get_weights())
        print(oculta5.get_weights())
        print(salida.get_weights())
    
    if opc == '4':
        salir = True
