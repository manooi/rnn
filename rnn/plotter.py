import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

class Plotter:
    """
    A class to handle plotting functionalities.
    """
    @staticmethod
    def plot_loss(history, title, num, params, save_path):
        """
        Plots the training and validation loss.
        """
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))

        plt.figure()
        plt.plot(epochs, loss, color='b', label='Train Loss')
        plt.plot(epochs, val_loss, color='r', label='Validation Loss')

        plt.title(title)
        plt.xlabel("epochs")
        plt.ylabel("MSE Parameter (MCM)")
        plt.legend()

        plt.text(0.74, 0.8, f'history = {params["history"]}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.74, 0.75, f'batch_size = {params["batch_size"]}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.74, 0.7, f'buffer_size = {params["buffer_size"]}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.74, 0.65, f'epochs = {params["epochs"]}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.74, 0.60, f'steps = {params["steps"]}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.74, 0.55, f'learning_rate = {params["add_learning_rate"]}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.74, 0.5, f'node = {params["node"]}', transform=plt.gca().transAxes, fontsize=10)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = f'case_{num}_RNN_Node_{params["node"]}_epochs_{params["epochs"]}_steps_{params["steps"]}_batch_size_{params["batch_size"]}_buffer_size_{params["buffer_size"]}_learning_rate_{params["add_learning_rate"]}_history_{params["history"]}_future_target_{params["future_target"]}.png'
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

    @staticmethod
    def plot_compare_value(dates, observed, predicted, hour, r_value, rmse_value, nse_value, save_path, params, num):
        """
        Plots the observed and predicted values over time.
        """
        plt.figure()
        plt.plot(dates, observed, linestyle='--', linewidth=0.75, color='blue', label='Observe')
        plt.plot(dates, predicted, color='red', label='Predict')
        plt.xlabel('Date')
        plt.ylabel('Discharge [cms]')
        plt.title('Predicted and Observe Values')
        plt.legend()
        plt.text(0.80, 0.8, f'hour = {hour + 1}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.80, 0.75, f'r = {r_value:.4f}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.80, 0.70, f'RMSE = {rmse_value:.4f}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.80, 0.65, f'NSE = {nse_value:.4f}', transform=plt.gca().transAxes, fontsize=10)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.grid(True)
        plt.ylim(0, 350)
        plt.gcf().autofmt_xdate()

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = f'case_{num}_Compare_t+{hour + 1}_RNN_node_{params["node"]}_epochs_{params["epochs"]}_steps_{params["steps"]}_batch_size_{params["batch_size"]}_buffer_size_{params["buffer_size"]}_learning_rate_{params["add_learning_rate"]}_history_{params["history"]}_future_target_{params["future_target"]}.png'
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

    @staticmethod
    def plot_scatter_separate_hours(observed, predicted, hour, r_value, rmse_value, nse_value, save_path, params, num):
        """
        Creates scatter plots of observed vs. predicted values for each hour.
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(observed, predicted, marker='o', s=60, c='red', edgecolors='red', lw=0.5, alpha=0.8)
        plt.xlim(0, 350)
        plt.ylim(0, 350)
        plt.plot([0, 350], [0, 350], linestyle='--', color='black')
        plt.xlabel('Discharge Observed ,cms.')
        plt.ylabel('Discharge Predicted ,cms.')
        plt.title(f'Scatter Plot for hour {hour + 1}')
        plt.legend()
        plt.text(0.05, 0.9, f'hour = {hour + 1}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.05, 0.85, f'r = {r_value:.4f}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.05, 0.8, f'RMSE = {rmse_value:.4f}', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.05, 0.75, f'NSE = {nse_value:.4f}', transform=plt.gca().transAxes, fontsize=10)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = f'case_{num}_scatter_t+{hour + 1}_RNN_node_{params["node"]}_epochs_{params["epochs"]}_steps_{params["steps"]}_batch_size_{params["batch_size"]}_buffer_size_{params["buffer_size"]}_learning_rate_{params["add_learning_rate"]}_history_{params["history"]}_future_target_{params["future_target"]}.png'
        plt.savefig(os.path.join(save_path, filename))
        plt.close()