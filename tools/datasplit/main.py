from pathlib import Path
import random
import shutil


Proportion = [6,1]
if __name__ == "__main__":
    source_path = Path('datasets')
    save_path = Path('result')
    (save_path / 'images' / 'train').mkdir(parents=True)
    (save_path / 'labels' / 'train').mkdir(parents=True)
    (save_path / 'images' / 'val').mkdir(parents=True)
    (save_path / 'labels' / 'val').mkdir(parents=True)


    for dataclass in source_path.iterdir():

        # rename the label file
        for original in (dataclass/'label').glob('*.bmp'):
          new = original.parent / ((original.stem).split('_')[0]+'.bmp')
          shutil.move(original , new) # image

        image_list = [p.stem for p in (dataclass).glob('*.bmp')]
        k = len(image_list) // sum(Proportion) * Proportion[0]
        train_image_list = random.sample(image_list, k=k)
        print(f'{dataclass.name}:{k}')

        for data in dataclass.glob('*.bmp'):
            move_to = ('train' if data.stem in train_image_list else 'val')
            shutil.copy(data, save_path / Path('images') / move_to /data.name)
            label_path =  dataclass / 'label' / data.name
            shutil.copy(label_path, save_path / Path('labels') / move_to /data.name)