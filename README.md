# DegiroSmartPortfolio

---

## Overview

DegiroSmartPortfolio è uno script che fornisce previsioni a lungo termine sul portafoglio di investimenti importato da **Degiro**. Utilizzando reti **LSTM** (Long Short-Term Memory), lo script analizza i dati storici degli asset nel portafoglio e genera previsioni sui prezzi futuri per i prossimi **10 anni**. 

Inoltre, lo script gestisce automaticamente gli **ETF**: se non riesce a trovare i dati relativi a un **ISIN**, cerca l'indice replicato e utilizza i dati dell'indice per generare previsioni.

### Funzionalità principali
1. **Importazione del file CSV da Degiro**: Lo script legge il portafoglio di investimenti esportato da Degiro in formato CSV.
2. **Previsioni con LSTM**: Utilizza un modello LSTM per prevedere i prezzi futuri di ciascun asset per i prossimi **10 anni** (2520 giorni lavorativi).
3. **Gestione automatica degli ETF**: Se non trova dati per un ETF, cerca l'indice replicato tramite **Yahoo Finance**.
4. **Generazione di report HTML**: Crea un report HTML unico con i dettagli di ogni asset, incluso il nome, l'ISIN, il valore corrente e un grafico delle previsioni.
5. **Valore totale del portafoglio**: Mostra il valore complessivo attuale del portafoglio in EUR nella parte superiore del report.

---

## Requisiti

Per eseguire lo script, sono necessarie le seguenti librerie:

### Librerie Python
- **pandas**: Per la gestione e l'elaborazione dei dati.
- **numpy**: Per le operazioni numeriche.
- **yfinance**: Per ottenere i dati storici degli asset.
- **yahooquery**: Per cercare informazioni sugli ETF e sugli indici replicati.
- **scikit-learn**: Per normalizzare i dati.
- **tensorflow**: Per costruire e addestrare il modello LSTM.
- **plotly**: Per generare grafici interattivi.
- **jinja2**: Per creare e formattare il report HTML.

Puoi installare tutte le dipendenze con il seguente comando:

```bash
pip install pandas numpy yfinance yahooquery scikit-learn tensorflow plotly jinja2
```

---

## Utilizzo

1. **Preparazione del file CSV**
   - Esporta il tuo portafoglio Degiro in formato CSV, assicurandoti che includa almeno le seguenti colonne:
     - `Asset`: Il nome dell'asset.
     - `Code`: Il codice ISIN dell'asset.
     - `Value in EUR`: Il valore corrente dell'asset in EUR.

2. **Esecuzione dello script**
   - Esegui lo script specificando il file CSV del portafoglio Degiro:
   
     ```bash
     python degirosmartportfolio.py
     ```

   - Lo script leggerà il file CSV, calcolerà il valore totale del portafoglio e genererà previsioni per ogni asset o indice replicato.

3. **Output**
   - Lo script produrrà un **report HTML** chiamato `report.html`, che contiene:
     - Il valore totale del portafoglio in EUR.
     - I dettagli di ciascun asset (nome, ISIN, valore corrente in EUR).
     - Un grafico delle previsioni sui prezzi futuri per i prossimi 10 anni.

---

## Modello LSTM

Il modello **LSTM** utilizzato nello script prevede i prezzi futuri degli asset in base ai dati storici ottenuti da **Yahoo Finance**. È costituito da due strati **LSTM**, ciascuno seguito da un livello di **Dropout** per evitare l'overfitting, e strati **Dense** finali per produrre l'output delle previsioni.

### Descrizione del modello LSTM

L'LSTM è una rete neurale ricorrente (RNN) specializzata per gestire sequenze temporali con dipendenze a lungo termine. La rete LSTM ha tre componenti principali, noti come **gate** (porte):

1. **Forget Gate** (Gate di dimenticanza):
   Questa porta decide quante informazioni mantenere dallo stato precedente:
   ```
   f_t = σ(W_f * [h_(t-1), x_t] + b_f)
   ```
   Dove:
   - `h_(t-1)` è lo stato nascosto precedente.
   - `x_t` è l'input corrente.
   - `W_f` e `b_f` sono i pesi e bias associati alla forget gate.
   - `σ` è la funzione sigmoide.

2. **Input Gate** (Gate di ingresso):
   Questa porta decide quali nuove informazioni aggiornare nello stato della cella:
   ```
   i_t = σ(W_i * [h_(t-1), x_t] + b_i)
   ```

3. **Output Gate** (Gate di uscita):
   Questa porta determina l'output basato sullo stato attuale della cella:
   ```
   o_t = σ(W_o * [h_(t-1), x_t] + b_o)
   ```

### Stato della cella:

Il nuovo stato della cella, `C_t`, viene calcolato come:
```
C_t = f_t * C_(t-1) + i_t * C̃_t
```
Dove `C̃_t` è il nuovo valore candidato, calcolato con una funzione tangente iperbolica.

L'output finale `h_t` è:
```
h_t = o_t * tanh(C_t)
```

### Raccomandazioni per l'addestramento
Anche se questo script è impostato per prevedere i prezzi basati sui dati storici, il modello **LSTM** potrebbe beneficiare di un ulteriore addestramento con parametri ottimizzati. Ecco alcuni suggerimenti:

- **Batch size**: Puoi sperimentare con batch size diversi. Uno dei valori comunemente utilizzati è 32, ma puoi anche provare 64 o 128.
- **Numero di epoche**: L'aumento del numero di epoche può migliorare l'apprendimento del modello, ma presta attenzione all'overfitting.
- **Normalizzazione dei dati**: Assicurati che i dati siano ben normalizzati per ottenere risultati accurati. Il **MinMaxScaler** è utilizzato in questo script per normalizzare i dati tra 0 e 1.

---

## Disclaimer

Questo progetto è creato esclusivamente a scopo **accademico e di apprendimento**. Le previsioni generate non devono essere utilizzate come strumento decisionale per investimenti reali. I modelli e le previsioni non sono accurati al 100%, e non mi assumo alcuna responsabilità per eventuali perdite derivanti dall'uso di queste previsioni.

Il modello LSTM è soggetto a limitazioni e incertezze, soprattutto per previsioni a lungo termine. Raccomandiamo sempre di fare affidamento su analisi professionali per decisioni finanziarie reali.

---

## Contributi

Se vuoi contribuire a migliorare questo progetto o proporre nuove funzionalità, sentiti libero di inviare una pull request o di aprire un'issue!

---

### Licenza

Questo progetto è distribuito sotto la licenza MIT.

---

Grazie per aver usato **DegiroSmartPortfolio**!

---

### Note finali
Se hai domande o feedback, non esitare a contattarmi.
