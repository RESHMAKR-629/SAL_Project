import kagglehub

# Download latest version
path = kagglehub.dataset_download("pranaykoppula/torgo-audio/F_Dys")

print("Path to dataset files:", path)