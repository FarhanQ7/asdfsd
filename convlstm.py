# Load the Pandas libraries with alias 'pd'
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
from random import randint, choice
import random
# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later)
#df = pd.read_csv("/Users/farhanqureshi/Downloads/Amazon.csv", usecols=["Dates" , "Open" , "Close" , "High" , "Low" , "Volume"])
filepath = "/Users/farhanqureshi/Downloads/Amazon.csv"


# Preview the first 5 lines of the loaded data
class stat:
    def __init__(self, filepath, starting_cash):
        self.bank = starting_cash
        self.asset = 0
        self.data  = pd.read_csv(filepath, usecols=["Dates" , "Open" , "Close" , "High" , "Low" , "Volume"])

    def buy(self,stock_price):
        amount = (.8)*(self.bank/stock_price)
        self.bank = self.bank - stock_price*amount
        self.asset = self.asset + stock_price*amount
        #print("bank: ")
        #print(self.asset)


    def sell(self,prev_price,stock_price):
        #print("asset: ")
        #print(self.asset)
        self.bank = self.bank + (self.asset/prev_price)*stock_price
        self.asset = 0



    def profit_cal(self):
        prof = self.bank + self.asset
        return prof

    def destroyer(self):
        self.bank = 0
        self.asset = 0



    # def simulate(self):
    #     for i in range(0, len(self.data)):
    #         if(len(self.data["Open"])> i+1):
    #
    #             if(self.data["Open"][i] < self.data["Open"][i+1]):
    #                 if(self.bank > self.data["Open"][i]):
    #                     self.buy(int(self.data["Open"][i]))
    #
    #             elif(self.data["Open"][i] > self.data["Open"][i+1]):
    #                 if(self.asset > 0):
    #                     if(i !=0):
    #                         self.sell(int(self.data["Open"][i-1]),int(self.data["Open"][i]))

    def return_df(self,step):
        frame = []
        frame.append(self.data["Open"][step])
        frame.append(self.data["Close"][step])
        frame.append(self.data["High"][step])
        frame.append(self.data["Low"][step])
        frame.append(self.data["Volume"][step])

        return frame

    def sim2(self,action,step,bool):
        if(action==1):
            self.buy(int(self.data["Open"][step]))

        elif(action==0):
             if(step !=0):

                 self.sell(int(self.data["Open"][step-1]),int(self.data["Open"][step]))
        #self.destroyer()
        
        return self.return_df(step), (self.profit_cal()/100000),bool,{}


    def sim3(self,action,step,bool):
        if(action==1):
            self.buy(int(self.data["Open"][step]))

        elif(action==0):
             if(step !=0):

                 self.sell(int(self.data["Open"][step-1]),int(self.data["Open"][step]))
        #self.destroyer()
        if(step == 52299):
            bool=True
        return self.return_df(step), (self.profit_cal()/100000),bool,{}








# starting_cash = input("Enter Starting Cash: ")
# starting_cash = int(starting_cash)
# att = stat(filepath,starting_cash)
# if(isinstance(starting_cash, int)):
#         for i in range(0,1100):
#             k = random.randint(0, 1)
#             print("profit: for descion: " +str(k)+" "+str(att.sim2(k,i,bool=False)))
#
#
#         print(f"final prof:{att.profit_cal()}" )
#
# else:
#     print("not int")

#print("Best Case Profit is: " + str(att.profit_cal()))

# att2 = stat()
# for i in range(0, len(df)):
#     if(len(df["Open"])> i+1):
#
#         if(df["Open"][i] < df["Open"][i+1]):
#             if(att2.bank > df["Open"][i]):
#                 att2.buy(int(df["Open"][i]))
#
#         elif(df["Open"][i] > df["Open"][i+1]):
#             if(att2.asset > 0):
#                 att2.sell(int(df["Open"][i-1]),int(df["Open"][i]))
#
#
#
#
# print("if 65% accuracy est- Profit is: " + str(att2.profit_cal()*.65))











# fig = go.Figure(data=[go.Candlestick(x=df['Dates'],
#                 open=df['Open'],
#                 high=df['High'],
#                 low=df['Low'],
#                 close=df['Close'])])
#
# fig.show()

# single_day = []
# units = []
#
# for i in range(0,391):
#     single_day.append(data["Open"][i])
#     units.append(i)
#
# print(single_day)
#
#
# def builder():
#     fig, ax = plt.subplots()
#     ax.plot(units, single_day)
#
#     ax.set(xlabel='units (s)', ylabel='price ($)',
#            title='Amazon Day: ' + str(i))
#     ax.grid()
#
#     fig.savefig("/Users/farhanqureshi/Desktop/amazon_seq/seq"+str(i) ".png", dpi=300)
