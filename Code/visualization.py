
from utils import *
def extract_prediction_from_heatmap(output_heatmap):
    """
    Given a single-channel heatmap tensor [1, H, W], find the (x,y) of its maximum.
    Returns (x, y) in pixel coordinates.
    """
    # output_heatmap: Tensor shape (1, H, W)
    hm = output_heatmap.squeeze(0)  # → (H, W)
    # flatten to find argmax
    idx = torch.argmax(hm)
    y = (idx // hm.size(1)).item()
    x = (idx %  hm.size(1)).item()
    return x, y

def visualize_predictions(model, loader, output_dir, radius=3,frame_info=3, device='cpu', early_stop=None):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    transform_to_pil = ToPILImage()
    with torch.no_grad():
        for i, (frames, targets) in enumerate(loader):
            # assume frames: [B, C, H, W] in [0,1] or [0,255], targets: [B, 3] (vis, x, y)
            frames = frames.to(device)
            outputs = model(frames)              # [B, C, H, W]
            # reduce across channels by mean or max to get a single heatmap:
            heatmaps = torch.argmax(F.softmax(outputs, dim=1), dim=1)  # [B, 1, H, W]
            for b in range(frames.size(0)):
                # --- prepare base image ---
                frame = frames[b].cpu()
                frame = frame[frame_info//2 * 3:(frame_info//2 +1)*3]
                # assume frame is normalized [0,1], convert to uint8 BGR
                img = np.array(transform_to_pil(frame)).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # --- draw ground-truth ball location ---
                vis, gt_x, gt_y = targets[b].cpu().tolist()
                if vis != 0:
                    cv2.circle(img, (int(gt_x), int(gt_y)), radius, (0,255,0), 2, lineType=cv2.LINE_AA)

                # save GT overlay
                gt_vis_path = os.path.join(output_dir, f"{i:04d}_{b:02d}_gt.png")
                cv2.imwrite(gt_vis_path, img)

                # --- draw predicted ball location on a fresh copy ---
                img_pred = img.copy()
                pred_x, pred_y = extract_prediction_from_heatmap(heatmaps[b])
                cv2.circle(img_pred, (pred_x, pred_y), radius, (0,0,255), 2, lineType=cv2.LINE_AA)
                pred_vis_path = os.path.join(output_dir, f"{i:04d}_{b:02d}_pred.png")
                cv2.imwrite(pred_vis_path, img_pred)

                # --- optionally, stitch side by side into one image for easy comparison ---
                both = np.hstack([img, img_pred])
                both_path = os.path.join(output_dir, f"{i:04d}_{b:02d}_comparison.png")
                cv2.imwrite(both_path, both)

            # stop after a handful if you want
            if early_stop:
                if i >= early_stop:
                    break
    return