import lstm
import time
import matplotlib.pyplot as plt
import argparse

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path',
                        help='data file .',
                        type=str,
                        default='./sp500.csv')
    args = parser.parse_args()
    path = args.data_path

    global_start_time = time.time()
    epochs  = 1
    seq_len= 50
    print('> Loading data... ')

    X_train, y_train, X_test, y_test = lstm.load_data(path, seq_len, True)
    print('> Data Loaded. Compiling...')
    model = lstm.build_model([1, 50, 100, 1])
    model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)
    predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    #predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    #predicted = lstm.predict_point_by_point(model, X_test)

    print('Training duration (s) : ', time.time() - global_start_time)
    plot_results_multiple(predictions, y_test, 50)
