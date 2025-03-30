import pandas as pd
import numpy as np
import os
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from rnn.data_processor import DataProcessor
from rnn.util import multivariate_data, nse
from rnn.plotter import Plotter
from rnn.rnn_model import RNNModel

class Runner:
    """
    A class to run the time series forecasting experiment with different parameters.
    """
    def __init__(self, config, path_config, csv_path):
        self.config = config
        self.num = 1
        self.csv_path = csv_path
        self.results = {"history": [], "future_target": [], "batch_size": [], "buffer_size": [],
                        "epochs": [], "steps": [], "add_learning_rate": [], "node": [],
                        **{f"r_t+{i+1}": [] for i in range(24)},
                        **{f"rmse_t+{i+1}": [] for i in range(24)},
                        **{f"nse_t+{i+1}": [] for i in range(24)}}
        self.path_config = path_config
        

    def run(self):
        """
        Iterates through the parameter grid and runs the experiment for each combination.
        """
        features_all = ['B.3A', 'B.10', 'B.8A', 'B.9', 'B.11']
        training_split = 70128
        validation_split = 78888
        testing_split = 96408 # Not directly used in the loop but defined

        data_processor = DataProcessor(self.csv_path, features_all, training_split, validation_split)
        dataset = data_processor.load_and_preprocess_data()
        data_mean, data_std = data_processor.get_standardization_params()
        y_train_target = data_processor.get_target_data_for_training()

        STEP = 1

        loss_plot_path = self.path_config["loss_plot_path"]
        compare_plot_path = self.path_config["compare_plot_path"]
        scatter_plot_path = self.path_config["scatter_plot_path"]
        parameter_save_path = self.path_config["parameter_save_path"]
        excel_save_path = self.path_config["excel_save_path"]


        for history in self.config['history']:
            for future_target in self.config['future_target']:
                for batch_size in self.config['batch_size']:
                    for buffer_size in self.config['buffer_size']:
                        for epochs in self.config['epochs']:
                            for steps in self.config['steps']:
                                for add_learning_rate in self.config['add_learning_rate']:
                                    for node in self.config['node']:
                                        params = {"history": history, "future_target": future_target,
                                                  "batch_size": batch_size, "buffer_size": buffer_size,
                                                  "epochs": epochs, "steps": steps,
                                                  "add_learning_rate": add_learning_rate, "node": node}

                                        mpl.rcParams['figure.figsize'] = (8, 6)
                                        mpl.rcParams['axes.grid'] = False

                                        x_train_ss, y_train_ss = multivariate_data(
                                            dataset, dataset[:, 1], 0, training_split, history, future_target, STEP)
                                        x_val_ss, y_val_ss = multivariate_data(
                                            dataset, dataset[:, 1], training_split, validation_split, history, future_target, STEP)
                                        x_test_ss, y_test_ss = multivariate_data(
                                            dataset, dataset[:, 1], validation_split, None, history, future_target, STEP)

                                        train_ss = tf.data.Dataset.from_tensor_slices((x_train_ss, y_train_ss))
                                        train_ss = train_ss.cache().shuffle(buffer_size).batch(batch_size).repeat()

                                        val_ss = tf.data.Dataset.from_tensor_slices((x_val_ss, y_val_ss))
                                        val_ss = val_ss.cache().batch(batch_size).repeat()

                                        val_steps = (validation_split - training_split + 1 - history) // batch_size
                                        test_steps = (len(dataset) - validation_split + 1 - history) // batch_size

                                        srnn_model = RNNModel.build_model(x_train_ss.shape[-2:], future_target, node)
                                        print(srnn_model.summary())
                                        srnn_multi_model_history = RNNModel.compile_and_fit(
                                            srnn_model, train_ss, val_ss, steps, epochs, val_steps, add_learning_rate)

                                        Plotter.plot_loss(srnn_multi_model_history, 'Multi-Step Training and validation loss', self.num, params, loss_plot_path)

                                        y_pred = srnn_model.predict(x_test_ss)
                                        predict_df = pd.DataFrame(y_pred)
                                        predict_df = (predict_df * data_std[1]) + data_mean[1]
                                        observe_df = pd.DataFrame(y_test_ss)
                                        observe_df = (observe_df * data_std[1]) + data_mean[1]

                                        if not os.path.isdir(parameter_save_path):
                                            os.makedirs(parameter_save_path)
                                        filename_weights = f'case_{self.num}_RNN_node_{node}_epochs_{epochs}_steps_{steps}_batch_size_{batch_size}_buffer_size_{buffer_size}_learning_rate_{add_learning_rate}_history_{history}_future_target_{future_target}.weights.h5'
                                        srnn_model.save_weights(os.path.join(parameter_save_path, filename_weights))

                                        r_values = []
                                        for i in range(predict_df.shape[1]):
                                            corr = predict_df[i].corr(observe_df[i])
                                            r_values.append(corr)

                                        rmse_values = []
                                        nse_values = []
                                        correlation_results = {}
                                        rmse_results = {}
                                        nse_results = {}

                                        for hour in range(predict_df.shape[1]):
                                            observed_values = observe_df.iloc[:, hour]
                                            predicted_values = predict_df.iloc[:, hour]
                                            correlation = observed_values.corr(predicted_values)
                                            rmse_value = np.sqrt(mean_squared_error(observed_values, predicted_values))
                                            nse_value = nse(observed_values.values, predicted_values.values)

                                            correlation_results[hour] = correlation
                                            rmse_results[hour] = rmse_value
                                            nse_results[hour] = nse_value
                                            rmse_values.append(rmse_value)
                                            nse_values.append(nse_value)

                                            dates = pd.to_datetime(pd.read_csv(self.csv_path)['Datetime']).iloc[training_split + history:training_split + history + len(observe_df)]
                                            Plotter.plot_compare_value(dates, observed_values, predicted_values, hour, r_values[hour], rmse_value, nse_value, compare_plot_path, params, self.num)
                                            Plotter.plot_scatter_separate_hours(observed_values, predicted_values, hour, r_values[hour], rmse_value, nse_value, scatter_plot_path, params, self.num)


                                        self.results["history"].append(history)
                                        self.results["future_target"].append(future_target)
                                        self.results["batch_size"].append(batch_size)
                                        self.results["buffer_size"].append(buffer_size)
                                        self.results["epochs"].append(epochs)
                                        self.results["steps"].append(steps)
                                        self.results["add_learning_rate"].append(add_learning_rate)
                                        self.results["node"].append(node)

                                        for i, r_val in enumerate(r_values):
                                            self.results[f"r_t+{i+1}"].append(r_val)
                                        for i, rmse_val in enumerate(rmse_values):
                                            self.results[f"rmse_t+{i+1}"].append(rmse_val)
                                        for i, nse_val in enumerate(nse_values):
                                            self.results[f"nse_t+{i+1}"].append(nse_val)

                                        self.num += 1

        df_results = pd.DataFrame(self.results)
        if not os.path.isdir(excel_save_path):
            os.makedirs(excel_save_path)
        df_results.to_csv(os.path.join(excel_save_path, f'RNN_{self.num - 1}_case.csv'), index=False)
        print(f'Done saving results to {os.path.join(excel_save_path, f"RNN_{self.num - 1}_case.csv")}')
