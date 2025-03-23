import torch
import json
import os
from datetime import datetime
import torch.nn as nn

class FocusedModel(nn.Module):
    """
    A modified model that uses a window of focus to select relevant parts of the input
    based on tokens in the prompt (could represent temporal or feature-wise focus).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, focus_window_size):
        super(FocusedModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.focus_window_size = focus_window_size
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, focus_window=[11136, 51]):
        try:
            # Ensure the values in focus_window are integers
            focus_window = [int(focus_window[0]), int(focus_window[1])]
        except ValueError as e:
            raise ValueError(f"Invalid value in focus_window: {focus_window}. Must be integers.") from e

        print(f"x shape before slicing: {x.shape}")
        
        # Handle slicing for 2D and 3D tensors
        if len(x.shape) == 2:  # If it's 2D
            x = x[:, focus_window[0]:focus_window[1]]
        elif len(x.shape) == 3:  # If it's 3D
            x = x[:, :, focus_window[0]:focus_window[1]]
        else:
            raise ValueError(f"Input tensor has unsupported shape: {x.shape}")
        
        print(f"x shape after slicing: {x.shape}")
        
        # Flatten the input to match the model architecture
        x = x.view(x.size(0), -1)  # Flatten to a 2D tensor (batch_size, features)
        
        # Feed the data through the layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def load_model(model_path, focus_window_size=5, device=None):
    """
    Load a PyTorch model from the specified path, handling both full models and state_dicts,
    incorporating a focus window for attention-based selection.

    Parameters:
    - model_path: Path to the model file.
    - focus_window_size: Size of the window of focus on the input.
    - device: Device to load the model on (CPU or CUDA). Defaults to 'cuda' if available.

    Returns:
    - PyTorch model loaded onto the specified device.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 Loading model from {model_path} on {device}...")

    # Load the state_dict and metadata
    try:
        with open(model_path.replace('_Master.pth', '_Master_metadata.json'), 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"❌ Metadata file for {model_path} not found!")
        return None
    except json.JSONDecodeError:
        print(f"❌ Failed to decode JSON in metadata file for {model_path}")
        return None

    input_dim = metadata.get('input_dim', None)
    hidden_dim = metadata.get('hidden_dim', None)
    output_dim = metadata.get('output_dim', None)
    
    if not all([input_dim, hidden_dim, output_dim]):
        print(f"❌ Incomplete metadata found for model: {model_path}")
        return None

    # Load the model data (either full model or state_dict)
    try:
        model_data = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"❌ Error loading model data from {model_path}: {e}")
        return None

    # Build the model with a focus window parameter
    model = FocusedModel(input_dim, hidden_dim, output_dim, focus_window_size)

    if isinstance(model_data, dict):  # If it's a state_dict
        print("📦 Model saved as state_dict. Loading state_dict into model...")

        new_state_dict = {}
        for k, v in model_data.items():
            if k.startswith("layers.0."):
                new_state_dict["fc1." + k.split("layers.0.")[1]] = v
            elif k.startswith("layers.2."):
                new_state_dict["fc2." + k.split("layers.2.")[1]] = v
            else:
                print(f"⚠️ Unexpected key in state_dict: {k}")

        try:
            model.load_state_dict(new_state_dict)
            print("✅ State_dict loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading state_dict: {e}")
            return None

    else:
        model = model_data

    model.to(device)
    model.eval()
    print(f"Model Architecture: {model}")
    print("✅ Model loaded successfully!")
    
    return model

def save_model(model, model_name, output_dir='./models'):
    """
    Save the PyTorch model with a datetime timestamp.
    
    Parameters:
    - model: The model to save.
    - model_name: The base name of the model.
    - output_dir: Directory to save the model.
    
    Returns:
    - Path to the saved model file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{model_name}_{timestamp}.pth"
    model_path = os.path.join(output_dir, filename)
    
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at: {model_path}")
    
    return model_path


def predict_features(model, audio_tensor, focus_window, device):
    """
    Predict features from the model with input validation.

    Parameters:
    - model: The model to use for prediction.
    - audio_tensor: The input audio tensor.
    - focus_window: The window size for inference.
    - device: The device to run the model on.

    Returns:
    - The predicted features as a numpy array.
    """
    # Ensure the tensor has the correct shape
    expected_shape = model.fc1.in_features  # Get model input shape dynamically
    if audio_tensor.shape[1] != expected_shape:
        raise ValueError(
            f"Invalid input shape: {audio_tensor.shape[1]}, expected {expected_shape}"
        )

    # Run inference
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        # Apply the focus window and get the model's predicted features
        predicted_features = model(audio_tensor, focus_window).cpu().numpy()  # Ensure it's on CPU for further processing

    return predicted_features
