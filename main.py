from runner import Runner

config = {
    "history": [25],
    "future_target": [24],
    "batch_size": [128],
    "buffer_size": [20000],
    "epochs": [200],
    "steps": [200],
    "add_learning_rate": [0.01],
    "node": [128],
}
path_config = {
    "loss_plot_path" : r"./result/pic/1_loss",
    "compare_plot_path" : r"./result/pic/2_compare",
    "scatter_plot_path" : r"./result/pic/3_scatter",
    "parameter_save_path" : r"./result/parameter",
    "excel_save_path" : r"./result/excel"
}

if __name__ == "__main__":
    csv_path = r"./training_data/raw_data.csv"
    r = Runner(config, path_config, csv_path)
    r.run()