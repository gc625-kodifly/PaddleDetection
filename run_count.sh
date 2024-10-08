#!/bin/bash

cam=$1
# cam2=$2
source /home/kodifly/PaddleDetection/paddle/bin/activate


declare -A camera_labels
camera_labels[0]=PCS-CAM-01
camera_labels[1]=PCS-CAM-02
camera_labels[2]=PCS-CAM-03
camera_labels[3]=PCS-CAM-04
camera_labels[4]=PCS-CAM-05

declare -A camera_device
camera_device[0]=0 # orin 1
camera_device[1]=1 # orin 1
camera_device[2]=0 # orin 2
camera_device[3]=1 # orin 2
camera_device[4]=2 # orin 2 



echo camera label ${camera_labels[${cam}]}
echo camera /dev/device${camera_device[${cam}]}



# pipeline=(
#   'v4l2src device=/dev/video'${camera_device[${cam}]}
#   '! video/x-raw, format=UYVY, width=3840, height=2160, framerate=30/1,'
#   'colorimetry=2:4:7:1, interlace-mode=progressive'
#   '! appsink sync=0 drop=1'
# )

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
pipeline=(
  'nvv4l2camerasrc device=/dev/video'${camera_device[${cam}]}
  '! video/x-raw(memory:NVMM), format=UYVY, width=3840, height=2160, framerate=30/1'
  '! nvvidconv'
  '! video/x-raw,format=(string)UYVY,width=1280, height=720'
  '! queue'
  '! appsink sync=0 drop=1'
)


# Join the array elements into a single string
pipeline_string="${pipeline[*]}"
# --rtsp 'v4l2src device=/dev/video'${camera_device[${cam}]}' ! video/x-raw, format=UYVY, width=3840, height=2160, framerate=30/1, colorimetry=2:4:7:1, interlace-mode=progressive ! appsink sync=0 drop=1' \

python /home/kodifly/PaddleDetection/deploy/pipeline/pipeline.py \
--config /home/kodifly/PaddleDetection/deploy/pipeline/config/infer_cfg_jetson${camera_device[${cam}]}.yml \
--rtsp "${pipeline_string}" \
--device=gpu \
--run_mode trt_fp16 \
--camera_label ${camera_labels[${cam}]} \
--dla_core ${camera_device[${cam}]} \
--do_entrance_counting --region_type=horizontal \
--enable_write_video \
--write_video_dir /mnt/data

# --do_break_in_counting --region_type=custom --region_polygon \
# 0 0 600 200 1280 520 1280 720 0 720


# --rtsp 'nvv4l2camerasrc device=/dev/video'${cam}' ! video/x-raw(memory:NVMM), format=UYVY, width=3840, height=2160, framerate=30/1, colorimetry=2:4:7:1, interlace-mode=progressive ! nvvidconv ! video/x-raw,format=UYVY ! appsink sync=0 drop=1' \
# --video_file /home/kodifly/Downloads/14-18-57.mp4 \
# --trt_calib_mode True \
# /home/kodifly/paddle/PaddleDetection/run_count.sh

# gst-launch-1.0 nvv4l2camerasrc device=/dev/video0 ! 'video/x-raw(memory:NVMM),format=UYVY,width=3840,height=2160,framerate=30/1' ! nvvidconv ! 'video/x-raw, format=(string)UYVY' ! queue ! autovideosink
