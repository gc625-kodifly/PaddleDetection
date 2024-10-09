#!/bin/bash

insmod /home/kodifly/SG4A_ORIN_GMSL2_Orin_YUVx4_JP5.1.2_L4TR35.4.1/ko/max96712.ko
insmod /home/kodifly/SG4A_ORIN_GMSL2_Orin_YUVx4_JP5.1.2_L4TR35.4.1/ko/sgx-yuv-gmsl2.ko

v4l2-ctl -d /dev/video0  -c sensor_mode=3
v4l2-ctl -d /dev/video1 -c sensor_mode=3
