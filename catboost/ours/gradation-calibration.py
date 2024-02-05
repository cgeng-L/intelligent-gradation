import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback
from catboost import CatBoostRegressor, Pool
import time

data = np.array([
    # put your training data here
])

X = data[::2]  
y = data[1::2]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Prepare the pool objects for CatBoost
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train) 
y_test_scaled = scaler_y.transform(y_test) 

# Initialize CatBoost Regressor
model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.1,
    depth=1,
    loss_function='MultiRMSE', 
    eval_metric='MultiRMSE',
    verbose=400, 
    early_stopping_rounds=None
)

start_training_time = time.time()
# Training the model
model.fit(
    X_train, y_train_scaled,
    eval_set=(X_test, y_test_scaled),
    use_best_model=True,
    plot=True
)

end_training_time = time.time()
training_time = end_training_time - start_training_time
print(f"Training time: {training_time:.2f} s")

start_evaluation_time = time.time()

n_trees = model.get_best_iteration() + 1 

metrics_df = pd.DataFrame(columns=['Epoch', 'Train_MAE', 'Train_MSE', 'Train_RMSE', 'Train_R2', 
                                   'Val_MAE', 'Val_MSE', 'Val_RMSE', 'Val_R2'])

for epoch in range(n_trees):
    y_pred_train_scaled = model.predict(X_train, ntree_end=epoch+1)
    y_pred_val_scaled = model.predict(X_test, ntree_end=epoch+1)

    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled)
    y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_pred_train)

    val_mae = mean_absolute_error(y_test, y_pred_val)
    val_mse = mean_squared_error(y_test, y_pred_val)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_test, y_pred_val)

    metrics_dict = {
        'Epoch': epoch,
        'Train_MAE': train_mae,
        'Train_MSE': train_mse,
        'Train_RMSE': train_rmse,
        'Train_R2': train_r2,
        'Val_MAE': val_mae,
        'Val_MSE': val_mse,
        'Val_RMSE': val_rmse,
        'Val_R2': val_r2
    }

    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics_dict, index=[0])], ignore_index=True)

# Save DataFrame to CSV
metrics_df.to_csv("catboost_training_validation_metrics.csv", index=False)

end_evaluation_time = time.time()
evaluation_time = end_evaluation_time - start_evaluation_time
print(f"Evaluation time: {evaluation_time:.2f} s")

# define true values

y_true_list = [
    # put true gradation here
]

y_true = np.array(y_true_list[0])

new_data_list = [
    # area gradation should be put here
]

new_data = np.array(new_data_list)[0]
# new_data_reshaped = new_data.reshape(1, -1)
# new_data_scaled = scaler_X.transform(new_data_reshaped)
# new_data_normalized = 100 * new_data / np.sum(new_data)

start_prediction_time = time.time()
predicted_scaled = model.predict(new_data)
predicted_scaled_reshaped = predicted_scaled.reshape(1, -1)
predicted_actual = scaler_y.inverse_transform(predicted_scaled_reshaped)

# Ensure all predicted values are non-negative
predicted_actual_clipped = np.clip(predicted_actual, 0, None)

# Normalize to ensure the sum is 100, making sure there are no negative values
predicted_normalized = 100 * predicted_actual_clipped / np.sum(predicted_actual_clipped)
predicted_normalized = predicted_normalized.flatten()

mae = mean_absolute_error(y_true, predicted_normalized)

mse = mean_squared_error(y_true, predicted_normalized, squared=True)

rmse = mean_squared_error(y_true, predicted_normalized, squared=False)

r2 = r2_score(y_true, predicted_normalized)

end_prediction_time = time.time()
prediction_time = end_prediction_time - start_prediction_time

print(f"Prediction time: {prediction_time:.6f} s")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")