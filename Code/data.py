from utils import *
class TrackNetDataset(Dataset):
    def __init__(self, base_path, frame_info=3, resize_size=(360,640), transform=None):
        super().__init__()
        self.base_path = base_path
        self.new_H, self.new_W = resize_size
        self.frame_info = frame_info
        self.transform = transform or transforms.Compose([
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor()
        ])

        self.data = []
        for game_name in sorted(os.listdir(base_path)):
            game_path = os.path.join(base_path, game_name)
            if not os.path.isdir(game_path):
                continue

            for clip_name in sorted(os.listdir(game_path)):
                clip_path = os.path.join(game_path, clip_name)
                if not os.path.isdir(clip_path):
                    continue

                label_path = os.path.join(clip_path, 'Label.csv')
                if not os.path.exists(label_path):
                    continue

                label_df = pd.read_csv(label_path)
                image_names = label_df['file name'].tolist()

                if len(image_names) < frame_info:
                    continue

                for idx in range(frame_info//2, len(image_names)-frame_info//2):
                    self.data.append({
                        'clip_path': clip_path,
                        'label_df': label_df,
                        'center_idx': idx,
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        clip_path = entry['clip_path']
        label_df = entry['label_df']
        center_idx = entry['center_idx']
        imgs = []
        for i in range(center_idx - self.frame_info//2, center_idx + self.frame_info//2 + 1):
            img_name = label_df.iloc[i]['file name']
            img_path = os.path.join(clip_path, img_name)
            img = Image.open(img_path).convert('RGB')
            if i == center_idx:
                orig_W, orig_H = img.size
            if self.transform: 
                img = self.transform(img)
            imgs.append(img)
           
        imgs = torch.cat(imgs, dim=0)  
        visibility = label_df.iloc[center_idx]['visibility']
        x = label_df.iloc[center_idx]['x-coordinate']
        y = label_df.iloc[center_idx]['y-coordinate']
        if math.isnan(x) or math.isnan(y):
            x_resized, y_resized = -1, -1
        else:
            x_resized = x * (self.new_W / orig_W)
            y_resized = y * (self.new_H / orig_H)

        target = torch.tensor([visibility, x_resized, y_resized], dtype=torch.float32)

        return imgs, target
