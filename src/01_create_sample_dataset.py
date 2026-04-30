from pathlib import Path
import random, csv
from PIL import Image, ImageDraw, ImageFilter
SEED=42
random.seed(SEED)
BASE_DIR=Path('data/sample_crop_dataset')
CLASSES={'healthy_leaf':(55,145,60),'leaf_blight':(130,110,45),'leaf_spot':(75,125,55)}
def draw_leaf(class_name,color,index):
    img=Image.new('RGB',(96,96),(235,245,230)); draw=ImageDraw.Draw(img)
    cx,cy=48+random.randint(-4,4),48+random.randint(-4,4)
    draw.ellipse((cx-28,cy-38,cx+28,cy+38),fill=color)
    draw.line((cx,cy-32,cx,cy+32),fill=(30,90,35),width=2)
    if class_name=='leaf_spot':
        for _ in range(9):
            x,y=random.randint(25,70),random.randint(20,75); r=random.randint(2,5)
            draw.ellipse((x-r,y-r,x+r,y+r),fill=(95,45,25))
    elif class_name=='leaf_blight':
        for _ in range(4):
            x1,y1=random.randint(20,55),random.randint(15,70); x2,y2=x1+random.randint(15,30),y1+random.randint(5,18)
            draw.polygon([(x1,y1),(x2,y1+8),(x2-5,y2),(x1-3,y2-6)],fill=(155,120,45))
    return img.filter(ImageFilter.SMOOTH)
def main():
    BASE_DIR.mkdir(parents=True,exist_ok=True); rows=[]
    for c,col in CLASSES.items():
        d=BASE_DIR/c; d.mkdir(parents=True,exist_ok=True)
        for i in range(40):
            p=d/f'{c}_{i:03d}.png'; draw_leaf(c,col,i).save(p)
            rows.append({'image_path':str(p),'label':c,'source':'milestone1_synthetic_sample'})
    with (BASE_DIR/'metadata.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['image_path','label','source']); w.writeheader(); w.writerows(rows)
    print(f'Created {len(rows)} sample images at {BASE_DIR}')
if __name__=='__main__': main()
