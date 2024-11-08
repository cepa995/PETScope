FROM ubuntu:22.04

# Install dependencies and clean up apt cache
RUN apt-get update && apt-get install -y curl unzip xvfb libx11-6 && \
    rm -rf /var/lib/apt/lists/*

# Copy the MATLAB Runtime ZIP file to the Docker image
# NOTE: Make sure the MATLAB runtime ZIP is in the same directory as this Dockerfile
COPY MATLAB_Runtime_R2022b_Update_10_glnxa64.zip /matlab_runtime_installer.zip

# Set DISPLAY for headless installation
ENV DISPLAY=:0

# Unzip and install MATLAB Runtime, and clean up temporary files
RUN unzip /matlab_runtime_installer.zip -d /opt/matlab_runtime_installer && \
    echo "\ndestinationFolder=/opt/matlab_runtime/v912" >> /opt/matlab_runtime_installer/matlabruntime_installer_input.txt && \
    echo "agreeToLicense=yes" >> /opt/matlab_runtime_installer/matlabruntime_installer_input.txt && \
    echo "outputFile=/opt/matlab_runtime_installer/installation.log" >> /opt/matlab_runtime_installer/matlabruntime_installer_input.txt && \
    /opt/matlab_runtime_installer/install -inputFile /opt/matlab_runtime_installer/matlabruntime_installer_input.txt && \
    rm -rf /matlab_runtime_installer.zip /opt/matlab_runtime_installer

# Download and install SPM12, and clean up temporary files
RUN curl -sSL https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/spm12/spm12_r7771_Linux_R2022b.zip -o spm12.zip && \
    unzip spm12.zip -d /opt && \
    rm spm12.zip

# Add MATLAB Runtime to the PATH and LD_LIBRARY_PATH
ENV PATH="/opt/matlab_runtime/v912/bin:/opt/spm12:$PATH"
ENV LD_LIBRARY_PATH="/opt/matlab_runtime/v912/runtime/glnxa64:/opt/matlab_runtime/v912/bin/glnxa64:/opt/matlab_runtime/v912/sys/os/glnxa64:/opt/matlab_runtime/v912/extern/bin/glnxa64"

WORKDIR /opt/spm12

# Set the ENTRYPOINT to ensure all commands start with the SPM12 script
ENTRYPOINT ["/opt/spm12/run_spm12.sh", "/opt/matlab_runtime/v912/R2022b"]
