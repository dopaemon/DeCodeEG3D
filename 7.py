#@title Save optimization video
from IPython.display import display, HTML
from skvideo.io import FFmpegWriter
from base64 import b64encode


def save_gif(frames, out_path="video.mp4", display=False):
    if display:
        tq = tqdm
    else:
        tq = lambda x: x
    writer = FFmpegWriter(out_path)
    for frame in tq(frames):
        writer.writeFrame(np.asarray(frame))
        if display:
            from IPython.display import display as show
            show(frame)
    writer.close()


#@markdown Set to 128 if the notebook crashes
resolution = 256  #@param {type: "integer"}
out_path = "video.gif"  #@param {type: "string"}
save_gif([f.resize((resolution, resolution)) for f in frames], out_path)
video = open(out_path, "rb").read()
data_url = f"data:image/gif;base64," + b64encode(video).decode()
display(HTML(f"{data_url}>"))
