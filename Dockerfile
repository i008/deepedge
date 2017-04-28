FROM continuumio/miniconda3
RUN conda install scikit-image
RUN conda install tensorflow=1.0
RUN pip install jupyter
RUN pip install boto3==1.4.4
RUN pip install imutils==0.4.2
RUN pip install Keras==2.0.3
RUN pip install scipy==0.19.0
RUN pip install numpy==1.12.1
RUN pip install matplotlib==2.0.0
RUN pip install h5py
RUN apt-get install libtcmalloc-minimal4
ENV LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"

RUN mkdir /de
WORKDIR /de
ADD unet2.keras /de
ADD detect_edges.py /de
ADD edge_notebook.ipynb /de
ADD TOY.jpg /de






