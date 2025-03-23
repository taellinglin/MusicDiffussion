import os
import torch
import plotly.graph_objects as go
import numpy as np


def load_model_parameters(model_path):
    """Load model parameters from a .pth file."""
    if not os.path.isfile(model_path):
        return f"❌ Model not found: {model_path}", None, None

    # Load the model's state_dict
    model_data = torch.load(model_path, map_location="cpu")

    # Extract parameters from the state_dict
    param_info = {
        "Total Parameters": sum(p.numel() for p in model_data.values()),
        "Layers": len(model_data),
        "Keys": list(model_data.keys())
    }

    layer_params = extract_layer_params(model_data)
    
    # ✅ Generate exactly 4 plots, using placeholders if necessary
    plots = visualize_3d_layers(model_data)

    # Fill with placeholders if fewer than 4 plots
    while len(plots) < 4:
        plots.append(create_placeholder_plot(f"Empty Layer {len(plots) + 1}"))

    return param_info, plots, layer_params


def extract_layer_params(model_data):
    """Extracts detailed parameter info per layer."""
    layers = []

    for name, params in model_data.items():
        layers.append({
            "name": name,
            "shape": list(params.shape),
            "num_params": params.numel(),
            "dtype": str(params.dtype)
        })

    return layers


def visualize_3d_layers(model_data):
    """Generate 3D visualizations of the model layers using Plotly."""
    plots = []

    for name, params in model_data.items():
        # Only visualize layers with 2D tensors
        if len(params.shape) != 2:
            continue

        # Convert tensor to NumPy for visualization
        data = params.detach().numpy()

        # 3D Surface Plot
        fig = go.Figure(data=[go.Surface(z=data)])

        fig.update_layout(
            title=f"{name}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Value",
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        plots.append(fig)

        # Stop after 4 visualizations
        if len(plots) >= 4:
            break

    return plots


def create_placeholder_plot(title="Placeholder"):
    """Create a blank 3D plot for missing layers."""
    fig = go.Figure()

    fig.add_trace(go.Surface(z=np.zeros((10, 10))))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Value",
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def adjust_layer_params(model_path, layer_params):
    """
    Modify and save the model parameters.
    
    Parameters:
    - model_path: Path to the .pth file
    - layer_params: List of layer modifications
    
    Returns:
    - Path to the saved model
    """
    if not os.path.isfile(model_path):
        return f"❌ Model not found: {model_path}"

    model = torch.load(model_path, map_location="cpu")

    # Apply modifications
    for layer in layer_params:
        layer_name = layer["name"]
        if layer_name in model:
            # Adjust parameters
            shape = tuple(layer["shape"])
            model[layer_name] = torch.randn(*shape)

    # Save the modified model
    new_model_path = model_path.replace(".pth", "_modified.pth")
    torch.save(model, new_model_path)

    return f"✅ Model saved at {new_model_path}"
