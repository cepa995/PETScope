FROM ubuntu:22.04

RUN mkdir /opt/bin /opt/lib /opt/include /opt/share
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y

# Add  ppa:deadsnakes repository to install Python 3.11
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa

# Install Python 3.11 and set it as the default version
RUN apt-get install -y g++ build-essential liblapack* git wget curl openssl cmake cmake-curses-gui vim \
    zlib1g-dev libssl-dev unzip python3.11 python3.11-dev python3.11-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# Add /opt/lib to library path
RUN echo "/opt/lib/" >> /etc/ld.so.conf.d/userLibraries.conf
RUN ldconfig

# Install Python dependencies
RUN pip install \
    antspyx==0.5.3 \
    click==8.1.7 \
    contourpy==1.3.0 \
    cycler==0.12.1 \
    joblib==1.4.2 \
    jsonpointer==2.1 \
    kiwisolver==1.4.7 \
    lxml==5.3.0 \
    matplotlib==3.9.2 \
    nibabel==5.2.1 \
    nilearn==0.10.4 \
    numpy==2.1.2 \
    openpyxl==3.1.5 \
    pandas==2.2.3 \
    patsy==0.5.6 \
    pillow==10.4.0 \
    pydicom==3.0.1 \
    pyparsing==3.1.4 \
    pytz==2024.2 \
    PyYAML==6.0.2 \
    scikit-learn==1.5.2 \
    scipy==1.14.1 \
    setuptools==72.1.0 \
    six==1.16.0 \
    statsmodels==0.14.4 \
    threadpoolctl==3.5.0 \
    tzdata==2024.2 \
    webcolors==24.8.0 \
    pytest==8.3.3 \
    pytest-dependency==0.6.0 \
    typer==0.12.5 \
    wheel==0.43.0

# Dynamic PET
RUN git clone https://github.com/cepa995/dynamicpet.git && \
    pip3 install -e dynamicpet
RUN echo "Dynamic PET path: $(which kineticmodel)"

# PETPVC
RUN wget https://github.com/UCL/PETPVC/releases/download/v1.2.4/PETPVC-1.2.4-Linux.tar.gz &&\
    tar -zxvf PETPVC-1.2.4-Linux.tar.gz &&\
    cp -r PETPVC-1.2.4/* /opt/ &&\
    rm -r PETPVC* 

# Convert3D (C3D)
RUN mkdir '/mnt/algo/tools' && \
    cd '/mnt/algo/tools' && \
    wget https://sourceforge.net/projects/c3d/files/c3d/Nightly/c3d-nightly-Linux-gcc64.tar.gz && \
    tar -zxvf c3d-nightly-Linux-gcc64.tar.gz && \
    mv c3d-1.4.2-Linux-gcc c3d-1.4.2 && \
    rm -rf c3d-nightly-Linux-gcc64.tar.gz
# Update the PATH environment variable to include the directory where the binary is located
ENV PATH="/mnt/algo/tools/c3d-1.4.2/bin:${PATH}"
# Ensure the binary has execution permissions
RUN chmod +x /mnt/algo/tools/c3d-1.4.2/bin/c3d
# Set the default command
CMD ["python"]