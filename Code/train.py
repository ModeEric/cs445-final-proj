from utils import *

def train(model, train_loader, optimizer, criterion, device='cuda', resize_size=(128, 256)):
    model.train()
    scaler = torch.GradScaler("cuda")
    running_loss = 0.0
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    running_loss = 0.0
    for iter_id, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.autocast("cuda"):
            outputs = model(images)
            gt = generate_heatmap(targets, H=resize_size[0], W=resize_size[1]).squeeze(1)
            loss = criterion(outputs, gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        batch_bar.set_postfix(
        loss="{:.04f}".format(running_loss/(iter_id+1)),
        lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()
    running_loss /= len(train_loader)
    batch_bar.close()
    return running_loss