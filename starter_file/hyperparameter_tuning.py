best_run = hdr.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()

print('Best run id:', best_run.id) 
print('\n Accuracy:', best_run_metrics['accuracy'])  
print('\n Learning rate:', best_run_metrics[r'learning_rate:'])  
print('\n Number of neurons in hidden layer', best_run_metrics[r'neurons:'])
print('\n Number of times the training over the dataset is run', best_run_metrics[r'epochs:'])
