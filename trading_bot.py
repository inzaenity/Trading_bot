import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import backtrader
class NeuralNetworkStrategy(backtrader):
    def __init__(self):
        # Download historical data for the SPX
        data = yf.download("SPX", start="2000-01-01", end="2020-12-31")

        # Preprocess data
        data = pd.DataFrame(data)
        data = data.dropna()

        # Split data into training, validation, and test sets
        train_data, test_data = train_test_split(data, test_size=0.2)
        val_data, test_data = train_test_split(test_data, test_size=0.5)

        # Define the neural network
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=train_data.shape[1], activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model on the training data
        self.model.fit(train_data, epochs=50, batch_size=32)

        # Evaluate the model on the validation data
        val_loss, val_acc = self.model.evaluate(val_data)
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

    def next(self):
        # Use the model to make predictions on the current data
        prediction = self.model.predict(self.data)

        # Use the prediction to make a trading decision
        if prediction > 0.5:
            self.buy()
        else:
            self.sell()

# Create an instance of the strategy
strategy = NeuralNetworkStrategy()

# Add the strategy to Backtrader
cerebro = Backtrader()
cerebro.addstrategy(strategy)

# Run the backtest
cerebro.run()
