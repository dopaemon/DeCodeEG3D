#@title Save mesh
from datetime import datetime


#@markdown Name of the file (leave empty to name automatically)
mesh_path = ""  #@param ["", "eg3d.ply"] {type: "string", allow-input: true}
#@markdown Path to save in google drive, leave empty if you want to store it in the Colab session only
drive_path = "EG3D"  #@param ["", "EG3D"] {type: "string", allow-input: true}
if not mesh_path:
    # TODO use hash of source image instead?
    mesh_path = f"e-{datetime.strftime(datetime.now(),'%Y_%m_%d-%H_%M_%S')}.ply"

#@markdown Download the resulting .obj file?
download_mesh = True  #@param {type: "boolean"}
kwargs = dict()  # TODO put something here
tri_mesh.export(mesh_path, **kwargs)

if download_mesh:
    from google.colab import files
    files.download(mesh_path)

if drive_path:
    from google.colab import drive
    print("Mounting drive...")
    drive.mount("/content/drive/")
    print("Saving to drive...")
    drive_path = f"/content/drive/MyDrive/{drive_path}"
    os.makedirs(drive_path, exist_ok=True)
    tri_mesh.export(f"{drive_path}/{mesh_path}", **kwargs)
