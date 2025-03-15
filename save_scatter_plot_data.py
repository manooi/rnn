import os
import pandas as pd

class ScatterPlotDataSaver:
    def __init__(self, num, hour, history, future_target, batch_size, buffer_size, EPOCHS, steps, add_learning_rate, Node):
        self.num = num
        self.hour = hour
        self.history  = history
        self.future_target = future_target
        self.batch_size  = batch_size
        self.buffer_size  = buffer_size
        self.EPOCHS  = EPOCHS
        self.steps  = steps
        self.add_learning_rate  = add_learning_rate
        self.Node = Node


    def get_file_name(self):
        return 'case_' + str(self.num)+ '_scatter_t+' + str(self.hour+1)+ '_RNN' + '_Node_' + str(self.Node)  + '_EPOCHS_' + str(self.EPOCHS)  + '_steps_' + str(self.steps) + '_batch_size_' + str(self.batch_size) + '_buffer_size_' + str(self.buffer_size) + '_learning_rate_' + str(self.add_learning_rate) + '_history_' + str(self.history) + '_future_target_' + str(self.future_target)

    def save(self, observe_df, predict_df, rmse_results, nse_results, r):
        path = "./scatter-plot-data"
        if not os.path.isdir(path):
            os.makedirs(path)

        # Prepare data to be saved in JSON
        data = {
            "hour": [self.hour],
            "observe_df": [observe_df.iloc[self.hour].values.tolist()],
            "predict_df": [predict_df.iloc[self.hour].values.tolist()],
            "rmse_results": [rmse_results[self.hour]],
            "nse_results": [nse_results[self.hour]],
            "r": [r],
        }

        filename = f"{path}/{self.get_file_name()}.csv"
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)


