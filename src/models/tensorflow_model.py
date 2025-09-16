# Modelo TensorFlow
"""
Modelo TensorFlow para previsão de métricas de campanhas
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class TensorFlowPredictor:
    """Classe para previsões usando TensorFlow/Keras"""

    def __init__(self, sequence_length=7, epochs=100, batch_size=32):
        """
        Inicializa o modelo TensorFlow

        Args:
            sequence_length (int): Comprimento da sequência de entrada
            epochs (int): Número de épocas de treinamento
            batch_size (int): Tamanho do batch
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def prepare_data(self, data: pd.DataFrame, target_column: str, date_column: str = 'date'):
        """
        Prepara os dados para o modelo TensorFlow

        Args:
            data (pd.DataFrame): Dados de entrada
            target_column (str): Coluna alvo para previsão
            date_column (str): Coluna de data
        """
        # Agrupa por data se necessário
        if data[date_column].duplicated().any():
            data = data.groupby(date_column)[target_column].sum().reset_index()

        # Ordena por data
        data = data.sort_values(date_column)

        # Extrai valores da coluna alvo
        self.data = data[target_column].values.reshape(-1, 1)

        # Normaliza os dados
        self.scaled_data = self.scaler.fit_transform(self.data)

    def create_sequences(self, data, sequence_length):
        """
        Cria sequências para treinamento do modelo

        Args:
            data (np.array): Dados normalizados
            sequence_length (int): Comprimento da sequência

        Returns:
            tuple: Sequências X e y
        """
        X, y = [], []

        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """
        Constrói o modelo LSTM

        Args:
            input_shape (tuple): Forma da entrada

        Returns:
            keras.Model: Modelo compilado
        """
        model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True,
                              input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])

        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def fit(self, validation_split=0.2):
        """
        Treina o modelo

        Args:
            validation_split (float): Proporção dos dados para validação

        Returns:
            dict: Métricas de treinamento
        """
        # Cria sequências
        X, y = self.create_sequences(self.scaled_data, self.sequence_length)

        if len(X) == 0:
            return {
                'error': 'Dados insuficientes para criar sequências',
                'success': False
            }

        # Reshape para LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Constrói modelo
        self.model = self.build_model((X.shape[1], 1))

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.001
        )

        # Treina modelo
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        # Calcula métricas finais
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]

        return {
            'final_loss': final_loss,
            'final_val_loss': final_val_loss,
            'epochs_trained': len(self.history.history['loss']),
            'success': True
        }

    def predict(self, steps: int = 7):
        """
        Faz previsões para os próximos períodos

        Args:
            steps (int): Número de períodos para prever

        Returns:
            pd.DataFrame: Previsões
        """
        if self.model is None:
            raise ValueError(
                "Modelo não foi treinado. Execute fit() primeiro.")

        # Usa os últimos dados para iniciar a previsão
        last_sequence = self.scaled_data[-self.sequence_length:]
        predictions = []

        current_sequence = last_sequence.copy()

        for _ in range(steps):
            # Reshape para previsão
            input_data = current_sequence.reshape((1, self.sequence_length, 1))

            # Faz previsão
            prediction = self.model.predict(input_data, verbose=0)
            predictions.append(prediction[0, 0])

            # Atualiza sequência
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = prediction[0, 0]

        # Desnormaliza previsões
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)

        # Cria DataFrame com previsões
        last_date = pd.Timestamp.now().date()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted': predictions.flatten()
        })

        return predictions_df

    def get_training_history(self):
        """
        Retorna histórico de treinamento

        Returns:
            dict: Histórico de loss e métricas
        """
        if self.history is None:
            return None

        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'mae': self.history.history['mae'],
            'val_mae': self.history.history['val_mae']
        }
