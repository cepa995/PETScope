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
RUN mkdir -p /mnt/algo/tools && \
    cd /mnt/algo/tools && \
    wget https://sourceforge.net/projects/c3d/files/c3d/Nightly/c3d-nightly-Linux-gcc64.tar.gz && \
    tar -zxvf c3d-nightly-Linux-gcc64.tar.gz && \
    mv c3d-1.4.2-Linux-gcc64 c3d-1.4.2 && \
    rm -rf c3d-nightly-Linux-gcc64.tar.gz

# Update the PATH environment variable to include the directory where the binary is located
ENV PATH="/mnt/algo/tools/c3d-1.4.2/bin:${PATH}"
# Ensure the binary has execution permissions
RUN chmod +x /mnt/algo/tools/c3d-1.4.2/bin/c3d

# Clone and install PETScope
RUN git clone https://github.com/cepa995/PETScope.git && \
    cd PETScope && \
    pip3 install -e .
ENTRYPOINT ["petscope"]
