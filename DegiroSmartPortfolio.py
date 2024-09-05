import pandas as pd
import numpy as np
import yfinance as yf
from yahooquery import Ticker
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objs as go
import plotly.io as pio
import os

# Funzione per ottenere l'indice replicato di un ETF tramite yahooquery
def get_replicating_index(etf_ticker):
    etf = Ticker(etf_ticker)
    summary_profile = etf.summary_profile.get(etf_ticker)
    if summary_profile and 'longBusinessSummary' in summary_profile:
        summary = summary_profile['longBusinessSummary']
        if 'replica' in summary or 'indice' in summary:
            print(f"Trovato l'indice replicato per {etf_ticker}: {summary}")
            return summary
        else:
            print(f"Non è stato possibile trovare l'indice replicato per {etf_ticker}.")
            return None
    else:
        print(f"Nessun profilo trovato per {etf_ticker}.")
        return None

# Funzione per preparare i dati per LSTM
def prepare_data(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Funzione per fare la previsione dei prezzi con LSTM
def predict_prices_lstm(asset, historical_data, prediction_days=2520):
    data = historical_data['Close'].values.reshape(-1, 1)

    # Normalizza i dati
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Dividi i dati in training e test (80% training)
    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 60:]

    # Prepara i dati per LSTM
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    # Ridimensiona i dati per LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Crea il modello LSTM migliorato
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Aggiungi Dropout per ridurre l'overfitting
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=50))
    model.add(Dense(units=1))

    # Compila il modello
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Allena il modello
    model.fit(X_train, y_train, batch_size=64, epochs=50)

    # Previsioni sui dati di test
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Previsioni future
    future_predictions = []
    last_data = scaled_data[-60:].reshape(1, -1)
    for _ in range(prediction_days):
        future_seq = np.array(last_data[:, -60:]).reshape(1, 60, 1)
        future_pred = model.predict(future_seq)
        future_predictions.append(scaler.inverse_transform(future_pred)[0][0])
        last_data = np.append(last_data, future_pred).reshape(1, -1)

    return np.array(future_predictions)

# Funzione per creare il grafico delle previsioni
def create_plot(asset_name, isin, current_value_eur, data, forecast, prediction_days=2520):
    fig = go.Figure()

    # Dati storici
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Prezzi storici'))

    # Previsioni
    future_dates = pd.date_range(data.index[-1], periods=prediction_days, freq='B')
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Previsioni'))

    fig.update_layout(
        title=f"Previsioni per {asset_name} (ISIN: {isin}) - Posizione corrente: {current_value_eur} EUR",
        xaxis_title="Data", yaxis_title="Prezzo di chiusura"
    )
    
    return fig

# Funzione per generare il report HTML
def generate_html_report(total_value, figures):
    html_content = f"""
    <html>
    <head>
        <title>Report Finanziario del Portafoglio</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px; }}
            h1 {{ text-align: center; color: #333; }}
            .container {{ max-width: 1200px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }}
            .total-value {{ font-size: 1.5em; margin-bottom: 20px; color: #4CAF50; text-align: center; }}
            .figure-container {{ margin-bottom: 40px; }}
            .figure-container h3 {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Report Finanziario del Portafoglio</h1>
            <p class="total-value">Valore Totale del Portafoglio: {total_value} EUR</p>
    """
    
    # Aggiungi tutti i grafici
    for figure_html in figures:
        html_content += f"""
        <div class="figure-container">
            {figure_html}
        </div>
        """
    
    # Chiusura del report
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content

# Funzione per salvare il report HTML
def save_html_report(filename, html_content):
    with open(filename, 'w') as f:
        f.write(html_content)
    print(f"Report HTML salvato come {filename}")

if __name__ == "__main__":
    # Carica il CSV Degiro pulito
    file_path = 'cleaned_degiro_portfolio.csv'
    portfolio_df = pd.read_csv(file_path)

    # Calcola il valore totale del portafoglio
    total_value = portfolio_df['Value in EUR'].sum()

    # Ottieni i codici ISIN, il nome dell'asset e la posizione corrente dal portafoglio Degiro
    assets = portfolio_df[['Asset', 'Code', 'Value in EUR']].to_dict(orient='records')

    # Lista per salvare i grafici in HTML
    figures_html = []

    # Ottieni i dati storici per 10 anni
    start_date = '2012-01-01'
    end_date = '2023-12-31'
    
    for asset in assets:
        asset_name = asset['Asset']
        isin = asset['Code']
        current_value_eur = asset['Value in EUR']

        stock = yf.Ticker(isin)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            # Cerca l'indice replicato dall'ETF se non troviamo dati
            index_info = get_replicating_index(isin)
            if index_info:
                index_ticker = index_info.split()[-1]
                stock = yf.Ticker(index_ticker)
                data = stock.history(start=start_date, end=end_date)
        
        if not data.empty:
            # Previsioni con LSTM
            forecast = predict_prices_lstm(asset_name, data, prediction_days=2520)

            # Crea il grafico con nome, ISIN e valore corrente
            fig = create_plot(asset_name, isin, current_value_eur, data, forecast)

            # Aggiungi il grafico in HTML
            figures_html.append(pio.to_html(fig, full_html=False))
        else:
            print(f"Nessun dato disponibile per {asset_name}, né per l'indice replicato.")

    # Genera il report HTML con il valore totale e tutti i grafici
    html_content = generate_html_report(total_value, figures_html)

    # Salva il report HTML
    save_html_report('report.html', html_content)

