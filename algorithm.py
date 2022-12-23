from AlgorithmImports import *
import tensorflow.compat.v1 as tf # attempt to disable TF error message
tf.disable_v2_behavior() # attempt to disable TF error message
graph = tf.get_default_graph()

import numpy as np
import statsmodels.api as sm
import random
from keras import backend as K 
from statsmodels.tsa.stattools import coint, adfuller
from QuantConnect.DataSource import *

# from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import gc
from matplotlib import pyplot as plt

from datetime import datetime, timedelta
tf.disable_eager_execution()

import pandas as pd
from copy import deepcopy

import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length):
        super(LSTM, self).__init__()
        self.num_layers = 2
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        # lstm, dropout, lstm, dropout, dense, dense
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        num_layers=self.num_layers, batch_first=True, dropout=0.2) # add dropout to all layers except last
        self.dropout = nn.Dropout(p=0.2)
        self.firstDense = nn.Linear(hidden_size, 5) #fully connected
        self.finalDense = nn.Linear(5, 1) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn[-1, :, :].view(-1, self.hidden_size) #reshaping the data for dropout+dense layers next
        out = self.dropout(hn)
        out = self.relu(out)
        out = self.firstDense(out) #first Dense
        out = self.relu(out)
        out = self.finalDense(out) #Final Output
        return out

class BasicTemplateAlgorithm(QCAlgorithm):

    def Initialize(self) -> None:

        #1. Required: Backtesting
        self.TrainingPeriod = "LT"

        #1. Required: Five years of backtest history
        if self.TrainingPeriod == "IS":
            self.SetStartDate(2017, 1, 1)
            self.SetEndDate(2021, 1, 1)
        if self.TrainingPeriod == "OOSA":
            self.SetStartDate(2022, 1, 1)
            self.SetEndDate(2022, 11, 1)
        if self.TrainingPeriod == "OOSB":
            self.SetStartDate(2016, 1, 1)
            self.SetEndDate(2017, 1, 1)
        if self.TrainingPeriod == "OOSC":
            self.SetStartDate(2010, 1, 1)
            self.SetEndDate(2011, 1, 1)
        if self.TrainingPeriod == "LT":
            self.SetStartDate(2022, 11, 9)
            self.SetEndDate(2022, 12, 9)
        if self.TrainingPeriod == "ST":
            self.SetStartDate(2020, 3, 1)
            self.SetEndDate(2020, 3, 31)
        #2. Required: Alpha Streams Models:
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
        #3. Required: Significant AUM Capacity
        self.SetCash(10000000)
        #4. Required: Benchmark to SPY, add the equity first
        self.bench = self.AddEquity("SPY", Resolution.Daily)
        self.SetBenchmark("SPY")
        self.lookback = 30
        self.long = "SPY"
        self.short = "SPY"
        self.models = []
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))
        self.SetExecution(ImmediateExecutionModel())

        # Set Scheduled Event Method For Our Model Retraining every month
        self.Schedule.On(self.DateRules.MonthStart(), 
            self.TimeRules.At(0, 0), 
            Action(self.BuildAllModels))

        # Set Scheduled Event Method For Our Model
        self.Schedule.On(self.DateRules.EveryDay(), 
            self.TimeRules.BeforeMarketClose("SPY", 5), 
            Action(self.EveryDayBeforeMarketClose))
        
        self.pairs = ["EWD", 
                        "EWO", 
                        "EWK",
                        "EWJ", 
                        "AAXJ", 
                        "EWZ", 
                        "EWI"]

        self.nameToIndex = {k: v for v, k in enumerate(self.pairs)}
        self.indexToName = {v: k for v, k in enumerate(self.pairs)}

        self.symbols = []
        for i in self.pairs:
            self.symbols.append(self.AddEquity(i, Resolution.Daily).Symbol)
        
        # Set our models
        self.bench.SetFeeModel(CustomFeeModel(self))
        self.bench.SetFillModel(CustomFillModel(self))
        self.bench.SetSlippageModel(CustomSlippageModel(self))
        self.bench.SetBuyingPowerModel(CustomBuyingPowerModel(self))

        # Algo Hyperparameters
        self.ibsEntry = 0.5
        self.ibsExit = 0.3
        self.lstmEntry = -0.2
        self.zscoreExit = 3.5

        # Build initial models
        self.BuildAllModels()

    def calcIBS(self, symbols):
        ibs = []
        for i in range(len(symbols)):
            security = self.Securities[symbols[i]]
            if security.High == security.Low:
                ibs.append(1)
            else:
                ibs.append((security.Close - security.Low) / (security.High - security.Low))
        return ibs

    def stats(self, symbols):
        
        #Use Statsmodels package to compute linear regression and ADF statistics
        self.df = self.History(symbols, self.lookback)
        self.dg = self.df['close'].unstack(level=0)
        
        ticker1= str(symbols[0])
        ticker2= str(symbols[1])

        Y = self.dg[ticker1].apply(lambda x: math.log(x))
        X = self.dg[ticker2].apply(lambda x: math.log(x))
        
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        sigma = math.sqrt(results.mse_resid) # Standard deviation of the residual
        slope = results.params[1]
        intercept = results.params[0]
        res = results.resid # Regression residual mean of res =0 by definition
        zscore = res/sigma
        adf = adfuller (res)
        return [adf, zscore, slope]

    def BuildGeneralModel(self, close):
        qb = self
        ### Preparing Data

        close = close.T

        # Scale data onto [0,1]
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        
        # Transform our data
        df = pd.DataFrame(self.scaler.fit_transform(close), index=close.index)
        
        # Feature engineer the data for input.
        input_ = df.iloc[1:]
        
        # Shift the data for 1-step backward as training output result.
        output = df.shift(-1).iloc[:-1]
        
        # Build feauture and label sets (using number of steps 60, and feature rank 1)
        features_set = []
        labels = []
        for i in range(60, input_.shape[0]):
            features_set.append(input_.iloc[i-60:i].values.reshape(-1, 1))
            labels.append(output.iloc[i])
        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

        ### Build Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = LSTM(input_size = 1, hidden_size = 50, seq_length = features_set.shape[1]).to(device)
        
        # Compile the model. We use Adam as optimizer for adpative step size and MSE as loss function since it is continuous data.
        loss_fn = nn.MSELoss()
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        epochs = 4
        batch_size = 70
        n_batches = int(features_set.shape[0] / batch_size)

        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")

            for i in range(n_batches):
                # Local batches and labels
                X, y = features_set[i*batch_size:(i+1)*batch_size,], labels[i*batch_size:(i+1)*batch_size,]
                X = torch.from_numpy(X).float()
                y = torch.from_numpy(y).float()

                pred = model(X)
                real = torch.from_numpy(np.array(y).flatten()).float()
                loss = loss_fn(pred, real)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 5 == 0:
                    loss, current = loss.item(), i
                    self.Debug(f"loss: {loss:.5f}  [{current:5d}/{n_batches:5d}]")
            
            # Since we're using SGD, we'll be using the size of data as batch number.
        return model

    def BuildAllModels(self):
        self.Debug("NEW MONTH -- BUILDING ALL MODELS "+str(self.Time))
        qb = self
        for model in self.models:
        #    K.clear_session()
        #    gc.collect
            del model
        self.models.clear()
        temp_models = []

        history = qb.History(qb.symbols, int(1.5*252), Resolution.Daily)
        close = history['close'].unstack(level=0)

        for securityName in qb.symbols:
            self.Debug("working on " + str(securityName))
            temp_models.append(self.BuildGeneralModel(pd.DataFrame([close[str(securityName)]])))
            temp_models[-1].eval()
            
        self.models = temp_models

    def EveryDayBeforeMarketClose(self) -> None:
        self.Debug(self.Time)
        qb = self

        # Use this to make algo decisions
        chart = [[str(sym.Value), -1, -1, -1, -1] for sym in qb.symbols] # symbol name, IBS, predicted price, actual price, normalization (respectively)

        # IBS
        ibsCalculations = self.calcIBS(qb.symbols)

        predictedPrices = []

        history = qb.History(qb.symbols, 60, Resolution.Daily)
        if history.empty: return

        close = history['close'].unstack(level=0)

        for idx, securityName in enumerate(qb.symbols):
            # Raw data transform
            unscaledDf = pd.DataFrame([close[str(securityName.Value)]]).T
            scaledDf = pd.DataFrame(self.scaler.transform(unscaledDf), index=unscaledDf.index)

            # Feature engineer the data for input
            input_ = []
            input_.append(scaledDf.values.reshape(-1, 1))
            input_ = np.array(input_)
            input_ = np.reshape(input_, (input_.shape[0], input_.shape[1], 1))
            input_ = torch.from_numpy(input_).float()

            # Prediction
            modelIndex = self.nameToIndex[str(securityName.Value)]
            # self.models[modelIndex]._make_predict_function()
            # self.models[modelIndex].eval() # already did eval in buildAllModels
            prediction = self.models[modelIndex](input_).detach().numpy()

            # Revert scaling to predicted price
            prediction = self.scaler.inverse_transform(prediction)
            predictedPrices.append(prediction)

            # Actual price
            actualPrice = qb.Securities[str(securityName.Value)].Price
            # self.Debug(str(securityName) + " : " + str(prediction) + " : "+ str(actualPrice))

            chart[idx][1] = ibsCalculations[idx]
            chart[idx][2] = prediction[0][0]
            chart[idx][3] = actualPrice
            chart[idx][4] = (prediction[0][0] - actualPrice) / actualPrice 

        chart.sort(reverse = True, key=lambda row: row[1])

        # Our choice of lost and short ETF
        self.pot_long = chart[-1]
        self.pot_short = chart[0]

        def valueAtRisk(self, long, short) -> bool:
            ''' Return True if exceed 95% confidence VaR threshold '''
            history = qb.History([long, short], 60, Resolution.Daily)
            df = history['close'].unstack(level = 0)
            
            dg = df.pct_change(1).dropna()
            dg_cov = dg.cov()

            if len(dg_cov.index) != 2:
                return False

            totalValue = qb.Portfolio.TotalPortfolioValue
            longValue = qb.Portfolio[long].AbsoluteHoldingsValue
            shortValue = qb.Portfolio[short].AbsoluteHoldingsValue
            longWeight = longValue / totalValue
            shortWeight = shortValue / totalValue

            wts = pd.DataFrame([longWeight, shortWeight])

            wts.index = dg_cov.index
            v = dg_cov.dot(wts)
            sigma = math.sqrt(wts.T.dot(v).values)
            
            zscore = 1.65 # 1 tailed, 95% confidence
            if zscore * sigma > 0.08:
                self.Debug("Value at Risk threshold met at " + str(zscore * sigma))
            return zscore * sigma > 0.08
            
        def getWeighting(self, sym1, sym2):
            results = self.stats([sym1, sym2])
            beta = results[2]
            zscore = results[1][-1]
            if beta > 0:
                return [1/(1+beta), beta/(1+beta), zscore]
            if beta <= 0:
                return [0, 1, 0]
        
        def findChart(chart, security):
            for x in chart:
                if x[0] == security:
                    return x
            return None

        use_weights = getWeighting(self, self.pot_long[0], self.pot_short[0])
        
        # If we are invested
        if self.Portfolio.Invested:
            chartLong = findChart(chart, self.long)
            chartShort = findChart(chart, self.short)
            if chartLong is None or chartShort is None:
                self.Liquidate()
            elif self.Portfolio[chartLong[0]].AbsoluteHoldingsValue != 0 and chartLong[1] >= 0.5+self.ibsExit and chartShort[1] <= 0.5-self.ibsExit: # Done mean-reverting
                self.Liquidate()
            elif self.Portfolio[chartLong[0]].AbsoluteHoldingsValue != 0 and abs(use_weights[2]) > self.zscoreExit: # ETFs have diverged forever
                self.Liquidate()
            elif (
                (self.pot_long[0] != self.long and self.pot_short[0] != self.short)
                and ((self.pot_long[1] < self.ibsEntry and self.pot_short[1] > self.ibsEntry) or (self.pot_long[4] >= -self.lstmEntry and self.pot_short[4] <= self.lstmEntry)) 
                or valueAtRisk(self, self.long, self.short)
                ): # Found new pair to trade
                self.Debug("Entering into fresh position")
                self.Liquidate()

        # If we are not invested
        elif (
            not self.Portfolio.Invested 
            and ((self.pot_long[1] < self.ibsEntry and self.pot_short[1] > self.ibsEntry) or (self.pot_long[4] >= -self.lstmEntry and self.pot_short[4] <= self.lstmEntry))
            ):
            self.Debug("Two securities: " + str(self.pot_long[0]) + " " + str(use_weights[0]) + " and " + str(self.pot_short[0]) + " " + str(use_weights[1]) + " " + str(self.pot_long[1]) + " " + str(self.pot_short[1]) + " " + str(self.pot_long[4]) + " " + str(self.pot_short[4]))
            self.SetHoldings(self.pot_long[0], use_weights[0]) 
            self.SetHoldings(self.pot_short[0], -use_weights[1])
            self.long = self.pot_long[0]
            self.short = self.pot_short[0]
        

class CustomFillModel(ImmediateFillModel):
    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm
        self.absoluteRemainingByOrderId = {}
        self.random = Random(387510346)

    def MarketFill(self, asset, order):
        absoluteRemaining = order.AbsoluteQuantity

        if order.Id in self.absoluteRemainingByOrderId.keys():
            absoluteRemaining = self.absoluteRemainingByOrderId[order.Id]

        fill = super().MarketFill(asset, order)
        absoluteFillQuantity = int(min(absoluteRemaining, self.random.Next(0, 2*int(order.AbsoluteQuantity))))
        fill.FillQuantity = np.sign(order.Quantity) * absoluteFillQuantity
        
        if absoluteRemaining == absoluteFillQuantity:
            fill.Status = OrderStatus.Filled
            if self.absoluteRemainingByOrderId.get(order.Id):
                self.absoluteRemainingByOrderId.pop(order.Id)
        else:
            absoluteRemaining = absoluteRemaining - absoluteFillQuantity
            self.absoluteRemainingByOrderId[order.Id] = absoluteRemaining
            fill.Status = OrderStatus.PartiallyFilled
        self.algorithm.Log(f"CustomFillModel: {fill}")
        return fill

class CustomFeeModel(FeeModel):
    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def GetOrderFee(self, parameters):
        # Custom fee math
        fee = max(1, parameters.Security.Price
                  * parameters.Order.AbsoluteQuantity
                  * 0.00001)
        self.algorithm.Log(f"CustomFeeModel: {fee}")
        return OrderFee(CashAmount(fee, "USD"))

class CustomSlippageModel:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def GetSlippageApproximation(self, asset, order):
        # Custom slippage math
        slippage = asset.Price * 0.00001 * np.log10(2*float(order.AbsoluteQuantity))
        self.algorithm.Log(f"CustomSlippageModel: {slippage}")
        return slippage

class CustomBuyingPowerModel(BuyingPowerModel):
    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def HasSufficientBuyingPowerForOrder(self, parameters):
        # Custom behavior: this model will assume that there is always enough buying power
        hasSufficientBuyingPowerForOrderResult = HasSufficientBuyingPowerForOrderResult(True)
        self.algorithm.Log(f"CustomBuyingPowerModel: {hasSufficientBuyingPowerForOrderResult.IsSufficient}")
        return hasSufficientBuyingPowerForOrderResult  
       
