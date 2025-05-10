from utils import *
def validate(model, val_loader, criterion, min_dist=5, device='cuda', resize_size=(128,256)):
    model.eval()
    losses = []
    total_dist = 0.0
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Validation')
    with torch.no_grad():
        for iter_id, (images, targets) in enumerate(val_loader):
            images = images.to(device)          # (B, 9, H, W)
            targets = targets.to(device)
            gt = generate_heatmap(targets,H=resize_size[0], W=resize_size[1]).squeeze(1)
            x_gt_batch = targets[:,1]  # (B,)
            y_gt_batch = targets[:, 2]  # (B,)
            vis_batch = targets[:, 0]   # (B,)

            outputs = model(images)                         # (B, 256, H, W)
            loss = criterion(outputs, gt)
            losses.append(loss.item())

            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)  # (B, H, W)
            B, H, W = preds.shape
            pred_classes_flat = preds.view(B, -1) # (B, HxW)
            max_indices = torch.argmax(pred_classes_flat, dim=1)  # (B,)
            y_pred = max_indices // W
            x_pred = max_indices % W
            # L2 distance between predicted and GT coordinates
            dist = torch.sqrt((x_pred.float() - x_gt_batch) ** 2 + (y_pred.float() - y_gt_batch) ** 2)
            # Logical mask
            pred_exists = (pred_classes_flat.max(dim=1).values > 0) 
            gt_exists = (vis_batch != 0)
            tp_mask = pred_exists & gt_exists & (dist < min_dist)
            fp_mask = pred_exists & (~gt_exists | (dist >= min_dist))
            fn_mask = (~pred_exists) & gt_exists
            tn_mask = (~pred_exists) & (~gt_exists)

            tp += tp_mask.sum().item()
            fp += fp_mask.sum().item()
            fn += fn_mask.sum().item()
            tn += tn_mask.sum().item()
            total_dist += torch.sum(dist)/B
            batch_bar.set_postfix(loss="{:.04f}".format(np.mean(losses)),
                                  dist="{:.04f}".format((total_dist/(iter_id+1)).item()),
                                  tp=tp, 
                                  tn=tn, 
                                  fp=fp, 
                                  fn=fn)
            batch_bar.update()
    eps = 1e-15
    total = tp + fp + fn + tn
    avg_loss = np.sum(losses) / total
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    batch_bar.close()
    return np.mean(losses), total_dist/len(val_loader), precision, recall, f1