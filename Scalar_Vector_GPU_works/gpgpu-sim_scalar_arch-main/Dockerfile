FROM ubuntu:14.04.5
RUN apt-get update -y
RUN apt-get install -y gcc g++ gcc-4.4 g++-4.4 make xutils bison flex doxygen python-pmw python-ply python-numpy libpng12-dev python-matplotlib doxygen graphviz git python-pmw python-ply python-numpy libpng12-dev libxi-dev libxmu-dev wget iproute2 expect xutils-dev libc-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libboost-dev libboost-system-dev libboost-filesystem-dev libboost-all-dev mpich libcuda1-304 binutils-gold vim
RUN wget http://developer.download.nvidia.com/compute/cuda/4_0/sdk/gpucomputingsdk_4.0.17_linux.run -P ~/.
RUN wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run -P ~/.
#RUN wget https://gist.githubusercontent.com/Kiran-r/a7ea775e5ddc0474754be3f234c6b5f1/raw/d7e5f8bb8e2dc7db22680eb50fab27100f8af325/cuda.exp -P ~/.
RUN wget https://gist.githubusercontent.com/wongdani/0dd68c6f834c20e2116d21367db58239/raw/9fadc90abdcc4b9edaec55ba95dfb28c4ed2b364/cuda-toolkit.exp -P ~/.
RUN wget https://gist.githubusercontent.com/wongdani/5571d1b15dd1791b6912ac6272cdfe5a/raw/a237475e44b92a26885c712eb6f82007c28a07d6/cuda-sdk.exp -P ~/.
RUN chmod 777 ~/gpucomputingsdk_4.0.17_linux.run ~/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run ~/cuda-toolkit.exp ~/cuda-sdk.exp
RUN echo "export CUDA_INSTALL_PATH=/usr/local/cuda" >> ~/.bashrc
RUN echo "export NVIDIA_COMPUTE_SDK_LOCATION=~/NVIDIA_GPU_Computing_SDK" >> ~/.bashrc
ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV NVIDIA_COMPUTE_SDK_LOCATION=~/NVIDIA_GPU_Computing_SDK
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.4 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.4 50
RUN git clone https://github.com/socal-ucr/gpgpu-sim_distribution ~/gpgpu-sim_distribution
RUN git clone https://github.com/socal-ucr/ispass2009-benchmarks ~/ispass2009-benchmarks
RUN expect ~/cuda-toolkit.exp
RUN expect ~/cuda-sdk.exp
RUN cd ~/NVIDIA_GPU_Computing_SDK/C/common; make
RUN cd ~/NVIDIA_GPU_Computing_SDK/shared; make
#RUN cd ~/NVIDIA_GPU_Computing_SDK/; make
RUN cd ~/ispass2009-benchmarks/; make -f Makefile.ispass-2009 common
RUN cp ~/NVIDIA_GPU_Computing_SDK/C/lib/libcutil* ~/common/lib/linux/
RUN cp ~/NVIDIA_GPU_Computing_SDK/shared/lib/libshrutil_x86_64.a ~/common/lib/linux/
RUN rm ~/cuda-sdk.exp ~/cuda-toolkit.exp ~/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run ~/gpucomputingsdk_4.0.17_linux.run
