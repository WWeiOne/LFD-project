import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMAResults
from arima_utils import print_and_save_AIC


class Arima_Model:
    def __init__(self, p_range, d_range, q_range, seasonal_m, test_only=False):
        # Generate all different combinations of p, q and q for Arima(p,d,q,m).
        self.pdq = list(itertools.product(p_range, d_range, q_range))

        # Generate all different combinations of P, D, Q and m for Arima(p,d,q)(P,D,Q,m) seasonal component.
        self.seasonal_pdq = [(p, d, q, seasonal_m)
                             for p, d, q in list(itertools.product(p_range, d_range, q_range))]
        
        if test_only: # use small search grid
            self.pdq= self.pdq[:2]
            self.seasonal_pdq= self.seasonal_pdq[:2]

    def fit(self, dataset):
        warnings.filterwarnings("ignore")
        # Grid Search to serach the lowerest AIC parameters combination.
        results_list = []
        for param in self.pdq:
            for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(dataset,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    print('ARIMA:', param, '; Seasonal:', param_seasonal, '; AIC:', results.aic)
                    results_list.append([param, param_seasonal, results.aic])
                except:
                    continue
        print_and_save_AIC(results_list)
        lowest_AIC_index = min(range(len(results_list)), key=lambda i: results_list[i][2])
        
        ## train the parameters with lowest AIC
        mod = sm.tsa.statespace.SARIMAX(dataset,
                                        order=results_list[lowest_AIC_index][0],
                                        seasonal_order=results_list[lowest_AIC_index][1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.model = mod.fit()
        self.model.save('arima.pickle')

    def load(self, model_name):
        self.model = ARIMAResults.load(model_name)

    def pred(self, dataset, plot_start, pred_start, steps):
        # get predict result
        pred_dynamic = self.model.get_prediction(
            start=pred_start, end=pred_start+steps-1, dynamic=True)

        # plot predict result from plot_start
        dataset = pd.Series(dataset)
        ax = dataset[plot_start:pred_start+steps].plot(label='observed', figsize=(15, 10))
        pred_index = range(pred_start, pred_start + steps)
        pd.Series(pred_dynamic.predicted_mean, index=pred_index).plot(label='Dynamic Forecast', ax=ax)
        
        #ax.fill_betweenx(ax.get_ylim(), plot_start, pred_start + steps,
        #                 alpha=.1, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel('multi-step forecast')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # return the predict data for comparasion across different models
        print(pred_start,pred_start + steps)
        print(len(dataset))
        return dataset[pred_start:pred_start + steps], pred_dynamic.predicted_mean