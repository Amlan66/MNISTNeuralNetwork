import torch
import pytest
from model import Net
from train import train_model
from torchsummary import summary

@pytest.fixture
def model():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return Net().to(device)

def test_parameter_count(model):
    """Test 1: Verify total parameters are less than 20000"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds 20000"

def test_model_accuracy(model, device):
    """Test 2: Verify model achieves > 99.40% accuracy"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Use CI mode but keep full 20 epochs
    final_accuracy = train_model(model, device, optimizer, 
                               epochs=20, ci_mode=True, return_accuracy=True)
    
    # Keep original accuracy threshold
    assert final_accuracy > 99.40, f"Model accuracy {final_accuracy} is less than 99.40%"

def test_batch_norm_usage(model):
    """Test 3: Verify BatchNorm is used and working"""
    # Check if model has BatchNorm layers
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model doesn't use Batch Normalization"
    
    # Check if BatchNorm is actually working during training
    model.train()
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for bn in bn_layers:
        assert bn.training, "BatchNorm layer is not in training mode"
        assert bn.track_running_stats, "BatchNorm is not tracking running stats"

def test_dropout_usage(model):
    """Test 4: Verify Dropout is used and working"""
    # Check if model has Dropout layers
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model doesn't use Dropout"
    
    # Check if Dropout is actually working during training
    model.train()
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for dropout in dropout_layers:
        assert dropout.training, "Dropout layer is not in training mode"
        assert dropout.p > 0, "Dropout probability is 0"

def test_gap_or_fc_usage(model):
    """Test 5: Verify GAP or FC layer is used"""
    # Check for GAP
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    # Check for FC
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    # Check for 1x1 conv as FC alternative
    has_1x1_fc = any(isinstance(m, torch.nn.Conv2d) and m.kernel_size == (1,1) 
                     for m in model.modules())
    
    assert has_gap or has_fc or has_1x1_fc, "Model doesn't use either GAP or FC layer"

def test_model_forward_pass(model):
    """Additional test: Verify forward pass works"""
    device = next(model.parameters()).device
    x = torch.randn(1, 1, 28, 28).to(device)
    output = model(x)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"

if __name__ == "__main__":
    pytest.main([__file__]) 