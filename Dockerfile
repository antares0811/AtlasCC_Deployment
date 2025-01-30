FROM python:3.9.20-slim
# Set working directory for installation
WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Clone and install nnUNet
COPY nnUNet22 ./nnUNet22

# Copy all necessary files and folders to /app
COPY static ./static
COPY check_pytorch.py ./check_pytorch.py
COPY api.py ./api.py
COPY STUNetTrainer_base_ft__nnUNetPlans__3d_fullres ./STUNetTrainer_base_ft__nnUNetPlans__3d_fullres
COPY templates ./templates

# Launch a bash shell (using recommended JSON format)
ENTRYPOINT ["python", "api.py"]