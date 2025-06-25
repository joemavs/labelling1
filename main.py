import torch
from sam2.build_sam import build_sam2_video_predictor
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backend_bases import MouseButton



# Load SAM2 model with specified config and checkpoint on CPU
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg,checkpoint,device="cpu")

video_dir = "videos/045_R"  # Directory of frames

# Function to display a segmentation mask on a Matplotlib axis
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.show()

# Function to visualize clicked points on the image
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


# Script to convert files to jpeg
def pngtojpeg():
    save_dir = "jpegs"  # Define name of jpeg dir
    target_dir = os.path.join(video_dir, save_dir)  # Creates path to save jpegs
    os.makedirs(target_dir, exist_ok=True)

    for file_name in os.listdir(video_dir):
        full_path = os.path.join(video_dir, file_name)

        # Skip if it's not a file (e.g. skip 'jpegs' directory)
        if not os.path.isfile(full_path):
            continue

        base_name = os.path.splitext(file_name)[0]  # gives "0" from "0.png"

        image = Image.open(full_path).convert("RGB")  # Replace with your actual image
        save_path = os.path.join(target_dir, f"{base_name}.jpg")
        image.save(save_path, "JPEG")
        print(f"Saved to: {save_path}")

png_video_dir = os.path.join(video_dir, "jpegs")

# Scan all the frame names
frame_names = [
    p for p in os.listdir(png_video_dir)
    if os.path.isfile(os.path.join(png_video_dir, p))
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

print(frame_names)

points = []
labels = []
def onclick(event):
    global point_coords, point_labels  # make them accessible outside
    if event.xdata is not None and event.ydata is not None:
        if event.button is MouseButton.LEFT:
            x,y, = int(event.xdata), int(event.ydata)
            print(f"Positive at: ({x},{y})")
            points.append([x,y])
            labels.append(1)
        elif event.button is MouseButton.RIGHT:
            x, y, = int(event.xdata), int(event.ydata)
            print(f"Negative at: ({x},{y})")
            points.append([x,y])
            labels.append(0)


# take a look at a specific frame
frame_idx: int = 828
fig, ax = plt.subplots(figsize=(10,10))
cid = fig.canvas.mpl_connect('button_press_event', onclick)
ax.set_title(f"frame {frame_idx}")
ax.imshow(Image.open(os.path.join(png_video_dir, frame_names[frame_idx])))
plt.show()

ann_frame_idx = 828  # The frame index we interact with
ann_obj_id = 1
inference_state = predictor.init_state(video_path=png_video_dir)

points = np.array(points,dtype=np.float32)
labels = np.array(labels,dtype=np.int32)

with torch.inference_mode():
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state = inference_state,
        frame_idx = ann_frame_idx,
        obj_id = ann_obj_id,
        points = points,
        labels = labels,
    )

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 10
plt.close("all") 
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)



