# Convolutional-network
Convolutional network basic code (Explained)



Table of Content:

1. Layers used to build a ConvNet
  1.1 Convolutional Layer.
  1.2 Pooling Layer.
  1.3 Normalization Layer.
  
2. Architecture

3. LSTM https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f

# Preparación del DATA ( Normalizar los atributos)

Scaler = MinMaxScaler( feature_range=(-1, 1) )
Scaled= scaler.fit_transform(series.values)
Series = pd.Dataframe(scaled)

# Se recomienda tamaño de pantalla 50 , cambio de la columna valor positivo ( hacia abajo)o negativo( hacia arriba) de acuerdo a necesidad para concatenar el Data 

window_size = 50

series_s = series.copy()
for i in range(window_size) :
series = pd.concat([series, series_s.shift(-(i+1))], axis = 1

series.dropna(axis=0, inplace=true)











----------------------------------------------------------------------------------------------------------------------------------------
1.1 Convolutional Layer







----------------------------------------------------------------------------------------------------------------------------------------
1.2 Pooling Layer







----------------------------------------------------------------------------------------------------------------------------------------
1.3 Normalization Layer









----------------------------------------------------------------------------------------------------------------------------------------
