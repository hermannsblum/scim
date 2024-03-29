FROM hermannsblum/nvidia-ros:noetic
SHELL ["/bin/bash", "-c"]
RUN . /opt/ros/noetic/setup.bash
RUN mkdir /ws
WORKDIR /ws
RUN catkin init \
  && catkin config --extend /opt/ros/noetic \
  && catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  && catkin config --merge-devel
RUN mkdir src
WORKDIR /ws/src
RUN wstool init \
  && git clone --branch hermann-devel https://github.com/ethz-asl/panoptic_mapping.git \
  && wstool merge panoptic_mapping/panoptic_mapping_https.rosinstall \
  && wstool update -j8 \
  && catkin build panoptic_mapping_utils point_cloud_io
RUN mkdir /semseg
WORKDIR /semseg
COPY ./deeplab ./deeplab
COPY ./setup.py ./setup.py
COPY ./semsegcluster ./semsegcluster
# create config if it does not exist
RUN if [ ! -f semsegcluster/settings.py ]; then echo "EXPERIMENT_STORAGE_FOLDER = '/sacred'\nTMPDIR = '/tmp'\nTMP_DIR =  '/tmp'\nEXP_OUT = '/outputs'" > semsegcluster/settings.py
# adapt config
RUN sed -i "s/EXP_OUT.*$/EXP_OUT = '/outputs'/g" semsegcluster/settings.py
RUN python3 -m pip install --ignore-installed .
WORKDIR /
