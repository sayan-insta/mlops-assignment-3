import joblib
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# 1. Load dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Load sklearn model
sk_model = joblib.load("model.joblib")
coef = sk_model.coef_
intercept = sk_model.intercept_

# 3. Save unquantized parameters
unquant_params = {"weights": coef, "bias": intercept}
joblib.dump(unquant_params, "unquant_params.joblib")

# 4. Manual quantization
weight_min = np.min(coef)
weight_max = np.max(coef)
scale = 255.0 / (weight_max - weight_min)
zero_point = weight_min

quant_weights = ((coef - zero_point) * scale).astype(np.uint8)
quant_bias = int((intercept - zero_point) * scale)

quant_params = {
    "weights": quant_weights,
    "bias": quant_bias,
    "scale": scale,
    "zero_point": zero_point
}
joblib.dump(quant_params, "quant_params.joblib")

# 5. Dequantize
dequant_weights = quant_weights.astype(np.float32) / scale + zero_point
dequant_bias = quant_bias / scale + zero_point

# 6. Define simple PyTorch model
class QuantizedLinearModel(torch.nn.Module):
    def __init__(self, weights, bias):
        super().__init__()
        self.fc = torch.nn.Linear(8, 1)
        self.fc.weight = torch.nn.Parameter(torch.tensor(weights.reshape(1, -1), dtype=torch.float32))
        self.fc.bias = torch.nn.Parameter(torch.tensor([bias], dtype=torch.float32))

    def forward(self, x):
        return self.fc(x)

# 7. Run inference with quantized model
X_tensor = torch.tensor(X_test, dtype=torch.float32)
quant_model = QuantizedLinearModel(dequant_weights, dequant_bias)
quant_preds = quant_model(X_tensor).detach().numpy().flatten()

# 8. R² Score & Model Size Comparison
original_preds = sk_model.predict(X_test)
r2_sklearn = r2_score(y_test, original_preds)
r2_quantized = r2_score(y_test, quant_preds)

size_unquant = os.path.getsize("unquant_params.joblib") / 1024  # in KB
size_quant = os.path.getsize("quant_params.joblib") / 1024  # in KB

print("\n Comparison:")
print(f"Original Sklearn Model R² Score: {r2_sklearn:.4f}")
print(f"Quantized PyTorch Model R² Score: {r2_quantized:.4f}")
print(f"Unquantized Params Size: {size_unquant:.2f} KB")
print(f"Quantized Params Size: {size_quant:.2f} KB")
