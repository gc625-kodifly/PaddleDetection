#!/bin/bash



echo loading max9295.ko
sudo insmod /home/kodifly/SG8A_ORIN_GMSL2X8_V2-AGX_YUV_JP5.1.2_L4TR35.4.1/ko/max9295.ko
echo loading max9296.ko
sudo insmod /home/kodifly/SG8A_ORIN_GMSL2X8_V2-AGX_YUV_JP5.1.2_L4TR35.4.1/ko/max9296.ko
echo sgx-yuv-gmsl2.ko
sudo insmod /home/kodifly/SG8A_ORIN_GMSL2X8_V2-AGX_YUV_JP5.1.2_L4TR35.4.1/ko/sgx-yuv-gmsl2.ko

echo v4l2-ctl on video 0
v4l2-ctl -d /dev/video0 -c sensor_mode=3,trig_pin=0xffff0007
echo v4l2-ctl on video 1
v4l2-ctl -d /dev/video1 -c sensor_mode=3,trig_pin=0xffff0007
echo v4l2-ctl on video 2
v4l2-ctl -d /dev/video2 -c sensor_mode=3,trig_pin=0xffff0007
