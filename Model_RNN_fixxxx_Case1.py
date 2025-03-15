# import packet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
import os
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
from save_scatter_plot_data import ScatterPlotDataSaver


i = 1
num = i

his_i = [25]
future_i = [24]
batch_i = [128]
buffer_i = [20000]
EPOCHS_i = [200]
steps_i = [200]
add_learning_rate_i = [0.01]
Node_i = [128]

# his_i = [1]
# future_i = [3]
# batch_i = [64]
# buffer_i = [10000]
# EPOCHS_i = [200]
# steps_i = [100]
# add_learning_rate_i = [0.00001,0.0001,0.001,0.01]
# Node_i = [16]




# his_i = [1]
# future_i = [3]
# batch_i = [32]
# buffer_i = [10000]
# EPOCHS_i = [10]
# steps_i = [100,150]
# add_learning_rate_i = [0.001]
# Node_i = [8]



# สร้าง dictionary เก็บค่า
parameter_grid = {
    "history": [],
    "future_target": [],
    "batch_size": [],
    "buffer_size": [],
    "EPOCHS": [],
    "steps": [],
    "add_learning_rate": [],
    "Node": [],
    "r_t+1": [],
    "r_t+2": [],
    "r_t+3": [],
    "r_t+4": [],
    "r_t+5": [],
    "r_t+6": [],
    "r_t+7": [],
    "r_t+8": [],
    "r_t+9": [],
    "r_t+10": [],
    "r_t+11": [],
    "r_t+12": [],
    "r_t+13": [],
    "r_t+14": [],
    "r_t+15": [],
    "r_t+16": [],
    "r_t+17": [],
    "r_t+18": [],
    "r_t+19": [],
    "r_t+20": [],
    "r_t+21": [],
    "r_t+22": [],
    "r_t+23": [],
    "r_t+24": [],
    "rmse_t+1": [],
    "rmse_t+2": [],
    "rmse_t+3": [],
    "rmse_t+4": [],
    "rmse_t+5": [],
    "rmse_t+6": [],
    "rmse_t+7": [],
    "rmse_t+8": [],
    "rmse_t+9": [],
    "rmse_t+10": [],
    "rmse_t+11": [],
    "rmse_t+12": [],
    "rmse_t+13": [],
    "rmse_t+14": [],
    "rmse_t+15": [],
    "rmse_t+16": [],
    "rmse_t+17": [],
    "rmse_t+18": [],
    "rmse_t+19": [],
    "rmse_t+20": [],
    "rmse_t+21": [],
    "rmse_t+22": [],
    "rmse_t+23": [],
    "rmse_t+24": [],
    "nse_t+1": [],
    "nse_t+2": [],
    "nse_t+3": [],
    "nse_t+4": [],
    "nse_t+5": [],
    "nse_t+6": [],
    "nse_t+7": [],
    "nse_t+8": [],
    "nse_t+9": [],
    "nse_t+10": [],
    "nse_t+11": [],
    "nse_t+12": [],
    "nse_t+13": [],
    "nse_t+14": [],
    "nse_t+15": [],
    "nse_t+16": [],
    "nse_t+17": [],
    "nse_t+18": [],
    "nse_t+19": [],
    "nse_t+20": [],
    "nse_t+21": [],
    "nse_t+22": [],
    "nse_t+23": [],
    "nse_t+24": [],
}

# วนซ้ำและเก็บค่าลงใน dictionary
for history in his_i:
    for future_target in future_i:
        for batch_size in batch_i:
            for buffer_size in buffer_i:
                for EPOCHS in EPOCHS_i:
                    for steps in steps_i:
                        for add_learning_rate in add_learning_rate_i:
                            for Node in Node_i:

                                mpl.rcParams['figure.figsize'] = (8,6) #กำหนดขนาดรูป
                                mpl.rcParams['axes.grid'] = False #กำหนดขนาดรูป

                                # import input data with csv file
                                csvpath = r"Raw_data.csv"
                                df2 = pd.read_csv(csvpath)
                                df2.head()

                                # select feature for model
                                features_all    = ['B.3A', 'B.10', 'B.8A', 'B.9', 'B.11'] # target is a second columns(inflow)
                                features        = df2[features_all]
                                features.index  = df2['Datetime']
                                features.head()
                                features.plot(subplots=True)

                                # standardize data
                                dataset     = features
                                data_mean   = dataset[:].mean(axis = 0)
                                data_std    = dataset[:].std(axis = 0)
                                dataset     = (dataset - data_mean)/ data_std
                                dataset     = dataset.values #change form dataframe to array

                                # defind range for training validation and testing
                                training_split      = 70128     #0-70127
                                validation_split    = 78888     #70128-78887
                                testing_split       = 96408     #78888-96408


                                X = pd.DataFrame(dataset)
                                Y = pd.DataFrame(dataset[0:training_split, 1])

                                #%% defind function
                                # function multivariate data
                                def multivariate_data(dataset , target , start_idx , end_idx ,
                                                      history_size ,target_size , step ,  single_step = False):
                                  data = []
                                  labels = []
                                  start_idx = start_idx + history_size
                                  if end_idx is None:
                                    end_idx = len(dataset)- target_size + 1
                                  for i in range(start_idx , end_idx ):
                                    idxs = range(i-history_size, i, step) # using step
                                    data.append(dataset[idxs])
                                    if single_step:
                                      labels.append(target[i+target_size-1])
                                    else:
                                      labels.append(target[i:i+target_size])
                                  return np.array(data) , np.array(labels)

                                def plot_loss(history, title,i,num,history1):

                                    history_dict = history.history

                                    loss = history.history['loss']
                                    val_loss = history.history['val_loss']
                                    epochs = range(len(loss))

                                    # Plot
                                    plt.figure()
                                    plt.plot(epochs, loss, color = 'b', label = 'Train Loss')
                                    plt.plot(epochs, val_loss, color = 'r', label = 'Validation Loss')

                                    # Set axes limit
                                    # plt.xlim(0 , EPOCHS)
                                    # plt.ylim(0 , 0.5)

                                    # Add title
                                    plt.title(title)

                                    # Add labels
                                    plt.xlabel("Epochs")
                                    plt.ylabel("MSE Parameter (MCM)")

                                    # show a legend on the plot
                                    plt.legend()

                                    # ระบุ พารามิเตอร์
                                    plt.text(0.74, 0.8, f'history = {history1}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.74, 0.75, f'batch_size = {batch_size}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.74, 0.7, f'buffer_size = {buffer_size}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.74, 0.65, f'EPOCHS = {EPOCHS}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.74, 0.60, f'steps = {steps}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.74, 0.55, f'learning_rate = {add_learning_rate}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.74, 0.5, f'Node = {Node}', transform=plt.gca().transAxes, fontsize=10)
                                    # plt.text(0.74, 0.55, f'Node2 = {Node2}', transform=plt.gca().transAxes, fontsize=10)
                                    # Display

                                    Out_path = r"./2_Progress/0_Case_1/Result/Pic/1_Loss"
                                    if not os.path.isdir(Out_path):
                                        os.makedirs(Out_path)

                                    plt.savefig(Out_path + '/case_' + str(num) + '_RNN' + '_Node_' + str(Node)  + '_EPOCHS_' + str(EPOCHS)  + '_steps_' + str(steps) + '_batch_size_' + str(batch_size) + '_buffer_size_' + str(buffer_size) + '_learning_rate_' + str(add_learning_rate) + '_history_' + str(history1) + '_future_target_' + str(future_target) +'.png')
                                    plt.close()
                                    # plt.show()



                                    num = num+1

                                #%%

                                # generate multivariate data
                                # history         = 1    #lookback 10 day as Tc
                                # future_target   = 3     #our targets will during 1 to 3 days in the future
                                STEP            = 1     #our observations will be sampled at one data point per hour

                                x_train_ss , y_train_ss = multivariate_data(dataset , dataset[:, 1],
                                                                            0, training_split, history,
                                                                            future_target, STEP)

                                x_val_ss , y_val_ss = multivariate_data(dataset , dataset[:,1] ,
                                                                        training_split , validation_split , history ,
                                                                        future_target, STEP)

                                x_test_ss , y_test_ss = multivariate_data(dataset , dataset[:,1] ,
                                                                          validation_split , None , history ,
                                                                          future_target, STEP)

                                print(x_train_ss.shape , y_train_ss.shape)


                                #%% # prepare tensorflow dataset
                                # batch_size = 128
                                # buffer_size = 12000

                                # tensorflow dataset
                                train_ss = tf.data.Dataset.from_tensor_slices((x_train_ss, y_train_ss))
                                train_ss = train_ss.cache().shuffle(buffer_size).batch(batch_size).repeat()

                                val_ss = tf.data.Dataset.from_tensor_slices((x_val_ss, y_val_ss))
                                val_ss = val_ss.cache().shuffle(buffer_size).batch(batch_size).repeat()

                                print(train_ss)
                                print(val_ss)

                                # This is how many steps to draw from `val_gen`
                                # in order to see the whole validation set:
                                val_steps = (validation_split - training_split+1 - history) // batch_size

                                # This is how many steps to draw from `test_gen`
                                # in order to see the whole test set:
                                test_steps = (len(dataset) - validation_split+1 - history) // batch_size

                                #%% # Modelling using Simple RNN

                                # EPOCHS = 200
                                # steps = 150


                                #Add learning rate
                                # add_learning_rate = 0.001  # Adjust the learning rate as needed

                                #RMSprop
                                # model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = add_learning_rate), loss='mae', metrics=['mae'])


                                SRNN_Multi_model = tf.keras.models.Sequential()
                                SRNN_Multi_model.add(tf.keras.layers.SimpleRNN(Node,input_shape = x_train_ss.shape[-2:], activation = 'relu'))
                                SRNN_Multi_model.add(tf.keras.layers.Dense(future_target))

                                print(SRNN_Multi_model.summary())

                                SRNN_Multi_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = add_learning_rate), loss='mae', metrics=['mae'])
                                SRNN_Multi_model_history = SRNN_Multi_model.fit(train_ss,
                                                                                          steps_per_epoch = steps,
                                                                                          epochs = EPOCHS,
                                                                                          validation_data = val_ss,
                                                                                          validation_steps = val_steps)

                                plot_loss(SRNN_Multi_model_history, 'Multi-Step Training and validation loss',i,num,history)


                                ## ไว้ load  parameter
                                ##SRNN_Multi_model.load_weights(r'C:\Users\pjjar\Desktop\Thesis_case1\RNN\Case 1\2_Progress\0_Case_1\Result 7 - use\Parameter\case_48_RNN_Node_128_EPOCHS_200_steps_200_batch_size_128_buffer_size_20000_learning_rate_0.01_history_25_future_target_24.weights.h5')

                                y_pred = SRNN_Multi_model.predict(x_test_ss)

                                predict_df = pd.DataFrame(y_pred)
                                predict_df = (predict_df * data_std[1] ) + data_mean[1]

                                observe_df = pd.DataFrame(y_test_ss)
                                observe_df = (observe_df * data_std[1] ) + data_mean[1]


                                Out_path_parameter = r"./2_Progress/0_Case_1/Result/Parameter"
                                if not os.path.isdir(Out_path_parameter):
                                    os.makedirs(Out_path_parameter)
                                ## ไว้เซฟ parameter
                                SRNN_Multi_model.save_weights(Out_path_parameter + '/case_' + str(num) + '_RNN' + '_Node_' + str(Node)  + '_EPOCHS_' + str(EPOCHS)  + '_steps_' + str(steps) + '_batch_size_' + str(batch_size) + '_buffer_size_' + str(buffer_size) + '_learning_rate_' + str(add_learning_rate) + '_history_' + str(history) + '_future_target_' + str(future_target) +'.weights.h5')



                                ## ไว้ load  parameter
                                ##SRNN_Multi_model.load_weights(r'C:\Users\BlackSword\OneDrive\master\Thesis\Paper\4. progress\3. model\tee\KEW_LOM.h5')



                                r = []
                                for i in range(len(predict_df.columns)):
                                    corr = predict_df[i].corr(observe_df[i])
                                    r.append(corr)


        #%%


                                # Define a function to calculate NSE
                                def nse(observed, predicted):
                                    return 1 - (np.sum((observed - predicted)**2) / np.sum((observed - np.mean(observed))**2))

                                # Initialize dictionaries to store results
                                correlation_results = {}
                                rmse_results = {}
                                nse_results = {}
                                rmse_re = []
                                nse_re = []


                                # Loop through the hours 0, 1, 2
                                for hour in range(24):
                                    # Extract the values for the specific day
                                    observed_values = observe_df.iloc[:, hour]
                                    predicted_values = predict_df.iloc[:, hour]
                                    # Calculate the correlation coefficient
                                    correlation = observed_values.corr(predicted_values)


                                    # Calculate RMSE
                                    rmse_value = np.sqrt(mean_squared_error(observed_values, predicted_values))
                                    rmse_re.append(rmse_value)
                                    # Calculate NSE
                                    nse_value = nse(observed_values.values, predicted_values.values)
                                    nse_re.append(nse_value)
                                    # Store the results in dictionaries
                                    correlation_results[hour] = correlation
                                    rmse_results[hour] = rmse_value
                                    nse_results[hour] = nse_value

                                # Display the results
                                for hour in range(24):
                                    print(f"Hour {hour+1}:")
                                    print(f"  Correlation: {correlation_results[hour]:.4f}")
                                    print(f"  RMSE: {rmse_results[hour]:.4f}")
                                    print(f"  NSE: {nse_results[hour]:.4f}")
                                    print()



                                def plot_compare_value(hour, size=None):

                                    # Set the figure size if size is specified
                                    if size:
                                        plt.figure(figsize=size)

                                    # Convert 'Datetime' column to datetime format if it's not already
                                    xx = df2['Datetime']
                                    ttt = type(xx)
                                    df2['Datetime'] = pd.to_datetime(df2['Datetime'])

                                    # Ensure you have the right range of dates
                                    # Get the dates for the range of the observed and predicted data
                                    start_date = df2['Datetime'].iloc[training_split + history].date()
                                    end_date = df2['Datetime'].iloc[training_split + history + len(observe_df)].date()

                                    # Select the dates for the range of your observation/prediction
                                    dates = df2['Datetime'].iloc[training_split + history:training_split + history + len(observe_df)]



                                    plt.plot(dates, observe_df[hour],linestyle = '--',linewidth = 0.75,color = 'blue', label='Observe')
                                    plt.plot(dates, predict_df[hour], color='red', label='Predict')
                                    plt.xlabel('Date')
                                    plt.ylabel('Discharge [cms]')
                                    plt.title('Predicted and Observe Values')
                                    plt.legend()
                                    plt.text(0.80, 0.8, f'hour = {hour+1}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.80, 0.75, f'r = {r[hour]:.4f}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.80, 0.70, f'RMSE = {rmse_results[hour]:.4f}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.80, 0.65, f'NSE = {nse_results[hour]:.4f}', transform=plt.gca().transAxes, fontsize=10)


                                # Configure x-axis to show dates spaced by 1 month
                                    ax = plt.gca()
                                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Set the interval of months on x-axis
                                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format date to Year-Month

                                    plt.grid(True)
                                    plt.ylim(0, 350)

                                    # Rotate date labels to prevent overlap
                                    plt.gcf().autofmt_xdate()




                                    # plt.grid(True)

                                    # #ตั้งค่าแกน Y
                                    # plt.ylim(0 , 50)


                                    # plt.show()

                                for hour in range(24):
                                    plt.figure()
                                    plot_compare_value(hour)
                                    # plt.savefig(f"comparison_day_{day+1}.png")  # Save each plot with a unique name
                                    Out_path3 = r"./2_Progress/0_Case_1/Result/Pic/2_Compare"
                                    if not os.path.isdir(Out_path3):
                                        os.makedirs(Out_path3)
                                    plt.savefig(Out_path3 + '/case_' + str(num)+ '_Compare_t+' + str(hour+1)+ '_RNN' + '_Node_' + str(Node)  + '_EPOCHS_' + str(EPOCHS)  + '_steps_' + str(steps) + '_batch_size_' + str(batch_size) + '_buffer_size_' + str(buffer_size) + '_learning_rate_' + str(add_learning_rate) + '_history_' + str(his_i) + '_future_target_' + str(future_target) +'.png')
                                    plt.close()

                                    # plt.show()


                                # # Save the plot
                                Out_path4 = r"./2_Progress/0_Case_1/Result/Pic/3_Scatter"







                                def plot_scatter_separate_hours(hour, observe_df, predict_df, rmse_results, nse_results, r,Out_path4):

                                    plt.figure(figsize=(6, 6))  # ขนาดกราฟ
                                    plt.scatter(observe_df[hour], predict_df[hour], marker='o', s=60, c='red', edgecolors='red', lw=0.5, alpha=0.8)
                                    plt.xlim(0, 350)
                                    plt.ylim(0, 350)
                                    # แกน 45 องศา
                                    plt.plot([0, 350], [0, 350], linestyle='--', color='black')
                                    # ใส่คำอธิบายแกน
                                    plt.xlabel('Discharge Observed ,cms.')
                                    plt.ylabel('Discharge Predicted ,cms.')
                                    plt.title(f'Scatter Plot for hour {hour+1}')
                                    plt.legend()
                                    plt.text(0.05, 0.9, f'hour = {hour+1}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.05, 0.85, f'r = {r[hour]:.4f}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.05, 0.8, f'RMSE = {rmse_results[hour]:.4f}', transform=plt.gca().transAxes, fontsize=10)
                                    plt.text(0.05, 0.75, f'NSE = {nse_results[hour]:.4f}', transform=plt.gca().transAxes, fontsize=10)
                                    # plt.show()

                                    # Out_path4 = r"C:\Users\BlackSword\OneDrive\master\Thesis\Paper\4. progress\3. model\use_case\case-1\5_station\result\pic\scatter"
                                    # plt.savefig(Out_path4 + '\case_' + str(num)+ '_scatter' + str(day)+ '_RNN' + '_Node_' + str(Node)  + '_EPOCHS_' + str(EPOCHS)  + '_steps_' + str(steps) + '_batch_size_' + str(batch_size) + '_buffer_size_' + str(buffer_size) + '_learning_rate_' + str(add_learning_rate) + '_history_' + str(his_i) + '_future_target_' + str(future_target) +'.png')
                                    # plt.close()  # Close the plot to avoid overlap
                                    
                                     
                                    
                                # เรียกใช้งานฟังก์ชันสำหรับแต่ละวัน
                                for hour in range(24):

                                    plt.figure()
                                    plot_scatter_separate_hours(hour, observe_df, predict_df, rmse_results, nse_results, r,Out_path4)
                                    # plt.figure()
                                    # plot_scatter_separate_days(day, observe_df, predict_df, rmse_re, nse_re, r)
                                    # plot_scatter_separate_days(day)
                                    # Save the plot
                                    # Show the plot
                                    # plt.show()
                                    # plt.tight_layout()
                                    Out_path4 = r"./2_Progress/0_Case_1/Result/Pic/3_Scatter"
                                    if not os.path.isdir(Out_path4):
                                        os.makedirs(Out_path4)
                                    plt.savefig(Out_path4 + '/case_' + str(num)+ '_scatter_t+' + str(hour+1)+ '_RNN' + '_Node_' + str(Node)  + '_EPOCHS_' + str(EPOCHS)  + '_steps_' + str(steps) + '_batch_size_' + str(batch_size) + '_buffer_size_' + str(buffer_size) + '_learning_rate_' + str(add_learning_rate) + '_history_' + str(his_i) + '_future_target_' + str(future_target) +'.png')
                                    plt.close()  # Close the plot to avoid overlap

                                    # plot_scatter_separate_days(observe_df, predict_df, r, rmse_re, nse_re, Out_path4)

                                    # plt.show()
                                    data_saver = ScatterPlotDataSaver(num, hour, history, future_target, batch_size, buffer_size, EPOCHS, steps, add_learning_rate, Node)
                                    data_saver.save(observe_df, predict_df, rmse_results, nse_results, r)

#%%




                                # for i in range(len(y_pred.columns)):
                                    # plot_compare_value(i)

                                parameter_grid["history"].append(history)
                                parameter_grid["future_target"].append(future_target)
                                parameter_grid["batch_size"].append(batch_size)
                                parameter_grid["buffer_size"].append(buffer_size)
                                parameter_grid["EPOCHS"].append(EPOCHS)
                                parameter_grid["steps"].append(steps)
                                parameter_grid["add_learning_rate"].append(add_learning_rate)
                                parameter_grid["Node"].append(Node)

                                parameter_grid["r_t+1"].append(r[0])
                                parameter_grid["r_t+2"].append(r[1])
                                parameter_grid["r_t+3"].append(r[2])
                                parameter_grid["r_t+4"].append(r[3])
                                parameter_grid["r_t+5"].append(r[4])
                                parameter_grid["r_t+6"].append(r[5])
                                parameter_grid["r_t+7"].append(r[6])
                                parameter_grid["r_t+8"].append(r[7])
                                parameter_grid["r_t+9"].append(r[8])
                                parameter_grid["r_t+10"].append(r[9])
                                parameter_grid["r_t+11"].append(r[10])
                                parameter_grid["r_t+12"].append(r[11])
                                parameter_grid["r_t+13"].append(r[12])
                                parameter_grid["r_t+14"].append(r[13])
                                parameter_grid["r_t+15"].append(r[14])
                                parameter_grid["r_t+16"].append(r[15])
                                parameter_grid["r_t+17"].append(r[16])
                                parameter_grid["r_t+18"].append(r[17])
                                parameter_grid["r_t+19"].append(r[18])
                                parameter_grid["r_t+20"].append(r[19])
                                parameter_grid["r_t+21"].append(r[20])
                                parameter_grid["r_t+22"].append(r[21])
                                parameter_grid["r_t+23"].append(r[22])
                                parameter_grid["r_t+24"].append(r[23])
                                parameter_grid["rmse_t+1"].append(rmse_re[0])
                                parameter_grid["rmse_t+2"].append(rmse_re[1])
                                parameter_grid["rmse_t+3"].append(rmse_re[2])
                                parameter_grid["rmse_t+4"].append(rmse_re[3])
                                parameter_grid["rmse_t+5"].append(rmse_re[4])
                                parameter_grid["rmse_t+6"].append(rmse_re[5])
                                parameter_grid["rmse_t+7"].append(rmse_re[6])
                                parameter_grid["rmse_t+8"].append(rmse_re[7])
                                parameter_grid["rmse_t+9"].append(rmse_re[8])
                                parameter_grid["rmse_t+10"].append(rmse_re[9])
                                parameter_grid["rmse_t+11"].append(rmse_re[10])
                                parameter_grid["rmse_t+12"].append(rmse_re[11])
                                parameter_grid["rmse_t+13"].append(rmse_re[12])
                                parameter_grid["rmse_t+14"].append(rmse_re[13])
                                parameter_grid["rmse_t+15"].append(rmse_re[14])
                                parameter_grid["rmse_t+16"].append(rmse_re[15])
                                parameter_grid["rmse_t+17"].append(rmse_re[16])
                                parameter_grid["rmse_t+18"].append(rmse_re[17])
                                parameter_grid["rmse_t+19"].append(rmse_re[18])
                                parameter_grid["rmse_t+20"].append(rmse_re[19])
                                parameter_grid["rmse_t+21"].append(rmse_re[20])
                                parameter_grid["rmse_t+22"].append(rmse_re[21])
                                parameter_grid["rmse_t+23"].append(rmse_re[22])
                                parameter_grid["rmse_t+24"].append(rmse_re[23])
                                parameter_grid["nse_t+1"].append(nse_re[0])
                                parameter_grid["nse_t+2"].append(nse_re[1])
                                parameter_grid["nse_t+3"].append(nse_re[2])
                                parameter_grid["nse_t+4"].append(nse_re[3])
                                parameter_grid["nse_t+5"].append(nse_re[4])
                                parameter_grid["nse_t+6"].append(nse_re[5])
                                parameter_grid["nse_t+7"].append(nse_re[6])
                                parameter_grid["nse_t+8"].append(nse_re[7])
                                parameter_grid["nse_t+9"].append(nse_re[8])
                                parameter_grid["nse_t+10"].append(nse_re[9])
                                parameter_grid["nse_t+11"].append(nse_re[10])
                                parameter_grid["nse_t+12"].append(nse_re[11])
                                parameter_grid["nse_t+13"].append(nse_re[12])
                                parameter_grid["nse_t+14"].append(nse_re[13])
                                parameter_grid["nse_t+15"].append(nse_re[14])
                                parameter_grid["nse_t+16"].append(nse_re[15])
                                parameter_grid["nse_t+17"].append(nse_re[16])
                                parameter_grid["nse_t+18"].append(nse_re[17])
                                parameter_grid["nse_t+19"].append(nse_re[18])
                                parameter_grid["nse_t+20"].append(nse_re[19])
                                parameter_grid["nse_t+21"].append(nse_re[20])
                                parameter_grid["nse_t+22"].append(nse_re[21])
                                parameter_grid["nse_t+23"].append(nse_re[22])
                                parameter_grid["nse_t+24"].append(nse_re[23])
                                i=i+1
                                num = num+1



                                # สร้าง DataFrame จาก dictionary
                                df = pd.DataFrame(parameter_grid)

                                Out_path2 = r"excel"
                                # df=pd.DataFrame(observe_df,date_test)
                                #พิม export_ชื่อไฟล์.csv
                                if not os.path.isdir(Out_path2):
                                    os.makedirs(Out_path2)


                                df.to_csv(Out_path2 + '/RNN_' + str(num-1)  + '_case' +'.csv')
                                print('Done'+'/RNN_' +'Obs2'+'.csv')


#%%นำข้อมูลออก
# Out_path2 = r"C:\Users\BlackSword\OneDrive\master\Thesis\Paper\4. progress\3. model\use_case\case-5-Hum-Tem\result\excel"
# # df=pd.DataFrame(observe_df,date_test)
# #พิม export_ชื่อไฟล์.csv


# df.to_csv(Out_path2 + '\RNN_' + str(num-1)  + '_case' +'.csv')
# print('Done'+'\RNN_' +'Obs2'+'.csv')