import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
import math


class ComplexityEvaluator:

    def __init__(self, nrow_samples, ncol_samples):
        self._nrow_samples = nrow_samples
        self._ncol_samples = ncol_samples

    def _time_samples(self, model, random_data_generator):
        rows_list = []
        for nrow in self._nrow_samples:
            for ncol in self._ncol_samples:
                train, labels = random_data_generator(nrow, ncol)

                start_time = time.time()
                model.fit(train, labels)
                elapsed_time = time.time() - start_time

                result = {"N": nrow, "P": ncol, "Time": elapsed_time}
                rows_list.append(result)

        return rows_list

    def Run(self, model, random_data_generator):
        data = pd.DataFrame(self._time_samples(model, random_data_generator))
        print(data)
        data = data.applymap(math.log)
        linear_model = LinearRegression(fit_intercept=True)
        linear_model.fit(data[["N", "P"]], data[["Time"]])
        return linear_model.coef_