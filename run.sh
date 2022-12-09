python LagrangePoint.py
cd Frames
ffmpeg -r 30 -pattern_type glob -i "*.jpg" -vcodec libx264 -s 1024x1024 -pix_fmt yuv420p axial_L45.mp4
