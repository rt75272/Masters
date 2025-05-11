import argparse
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torchvision import datasets, transforms # type: ignore
from torch.optim.lr_scheduler import StepLR # type: ignore
# ------------------------------------------------------------------------------------------------------------
# PyTorch MNIST Example.
#
# Image Classification Using ConvNets. This example demonstrates how to run image classification with 
# Convolutional Neural Networks ConvNets on the MNIST database.
#
# Usage:
#   $ python py_torch_demo.py
# ------------------------------------------------------------------------------------------------------------
class Net(nn.Module):
    """Define the neural network architecture with a class object."""
    def __init__(self):
        """Constructor to initialize the layers of the network."""
        super(Net, self).__init__() # Call the parent constructor.
        # Define layers: two convolutional layers, two dropout layers, and two fully connected layers.
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # First convolutional layer.
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # Second convolutional layer.
        self.dropout1 = nn.Dropout(0.25) # First dropout layer.
        self.dropout2 = nn.Dropout(0.5) # Second dropout layer.
        self.fc1 = nn.Linear(9216, 128) # First fully connected layer.
        self.fc2 = nn.Linear(128, 10) # Second fully connected layer.

    def forward(self, x):
        """Define the forward pass."""
        x = self.conv1(x) # Apply first convolution.
        x = F.relu(x) # Apply ReLU activation.
        x = self.conv2(x) # Apply second convolution.
        x = F.relu(x) # Apply ReLU activation.
        x = F.max_pool2d(x, 2) # Apply max pooling.
        x = self.dropout1(x) # Apply first dropout.
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layers.
        x = self.fc1(x) # Apply first fully connected layer.
        x = F.relu(x) # Apply ReLU activation.
        x = self.dropout2(x) # Apply second dropout.
        x = self.fc2(x) # Apply second fully connected layer.
        output = F.log_softmax(x, dim=1) # Apply log softmax for classification.
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    """Model training function."""
    model.train() # Set the model to training mode.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Move data to the specified device (CPU/GPU).
        optimizer.zero_grad() # Reset gradients.
        output = model(data) # Forward pass.
        loss = F.nll_loss(output, target) # Compute negative log-likelihood loss.
        loss.backward() # Backward pass.
        optimizer.step() # Update model parameters.
        if batch_idx % args.log_interval == 0: # Log progress at specified intervals.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run: # Exit early if dry-run mode is enabled.
                break

def test(model, device, test_loader):
    """Model testing function."""
    model.eval() # Set the model to evaluation mode.
    test_loss = 0
    correct = 0
    with torch.no_grad(): # Disable gradient computation for testing.
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # Move data to the specified device.
            output = model(data) # Forward pass
            test_loss += F.nll_loss(output, target, reduction='sum').item() # Sum up batch loss.
            pred = output.argmax(dim=1, keepdim=True) # Get the index of the max log-probability.
            correct += pred.eq(target.view_as(pred)).sum().item() # Count correct predictions.
    test_loss /= len(test_loader.dataset) # Compute average loss.
    # Print test results.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    """Main driver function to set up and run the training/testing process."""
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()
    # Check if accelerator (e.g., GPU) is available.
    use_accel = not args.no_accel and torch.accelerator.is_available()
    torch.manual_seed(args.seed) # Set random seed for reproducibility.
    # Set device to accelerator (if available) or CPU.
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    # Set up data loader parameters.
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)
    # Define data transformations.
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert images to tensors.
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize images.
    ])
    # Load MNIST dataset.
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs) # Training data loader.
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs) # Testing data loader.
    # Initialize the model, optimizer, and learning rate scheduler.
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # Training and testing loop.
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch) # Train the model.
        test(model, device, test_loader) # Test the model.
        scheduler.step() # Adjust learning rate.
    # Save the model if specified.
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

# The big red activation button.
if __name__ == '__main__':
    main()