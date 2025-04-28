# Enable error handling and verbose output
$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

# Define the path to Python executable and the script
$pythonPath = "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\python.exe"
$scriptPath = "Z:\Pix2Pix\PyTorch-Pix2Pix-Modified\modified_test.py"

# Define the arguments
$application_args = @(
    "--name pix7mask_resnet9_512",
    "--dataset_mode curtain",
    "--dataroot Z:\PelvisRongensDataset\dataset_split",
    "--model pix7mask",
    "--netG resnet_9blocks",
    "--load_size 256",
    "--crop_size 256",
    "--input_nc 2",
    "--output_nc 1"
)

$argsString = $application_args -join " "

# Execute the script with the arguments

# Start the process
Start-Process -FilePath $pythonPath -ArgumentList $scriptPath, $argsString -NoNewWindow -Wait
