# Enable error handling and verbose output
$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

# Define the path to Python executable and the script
$pythonPath = "C:\Users\alipa\AppData\Local\Microsoft\WindowsApps\python.exe"
$scriptPath = "C:\Users\alipa\repos\PyTorch-Pix2Pix-Modified\modified_train.py"

# Define the arguments
$application_args = @(
    "--dataset_mode reconstruction",
    "--dataroot C:\Users\alipa\repos\PyTorch-Pix2Pix-Modified\datasets\pelvis",
    "--name pix2pix_ivg_512",
    "--model pix2pix",
    "--netG aoe_ivg",
    "--display_id -1",
    "--display_freq 1",
    "--n_epochs 3",
    "--n_epochs_decay 1",
    "--load_size 256",
    "--crop_size 256",
    "--input_nc 1",
    "--output_nc 1",
    "--batch_size 4",
    "--gpu_ids -1"
)

$argsString = $application_args -join " "

# Execute the script with the arguments

# Start the process
Start-Process -FilePath $pythonPath -ArgumentList $scriptPath, $argsString -NoNewWindow -Wait
