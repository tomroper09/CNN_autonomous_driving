import Modelbag_copy
import os

# Choose GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Paths
root_path = "/home/alyjf10/self_driving_car/"                         # Root path
img_path = "training_data/training_data"                              # Training dataset name
csv_path = "combined_training_data.csv"                               # Training dataset CSV name
test_data = '/home/alyjf10/self_driving_car/data/test_data/test_data' # Test data path
test_data_csv = '/home/alyjf10/self_driving_car/data/test_data/test.csv' # Where to store the prediction result

# Initialize Model
all_model = {'DenseNet169', 'DenseNet121'} # Set of models to train

# Dictionary to store MSE results
mse_results = {}

for name in all_model:
    Model = Modelbag_copy.NN(model_name=name)
    # Load data
    # Model save path must be of type .h5!
    path = os.path.join('/home/alyjf10/self_driving_car/model', name)
    train_data, val_data = Model.load_data(root_path, img_path, csv_path, batch_size=32)
    Model.training(train_data, val_data, epochs=500, trained_model=None, model_save_path=path)
    
# Uncomment the line below to make predictions using the trained model
# Model.predict_model(trained_model_path='/home/alyjf10/self_driving/model/norm_1.h5', image_path=test_data, output_path=test_data_csv)

# After base_model

# Train three models, choose the best one, and then train again using the entire original dataset
