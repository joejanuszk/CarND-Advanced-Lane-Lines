from moviepy.editor import VideoFileClip
clip = VideoFileClip('../project_video.mp4')
subclip = clip.subclip(0, 10)
subclip.write_videofile('../project_video_output_test_2.mp4', fps=24, codec='mpeg4')
