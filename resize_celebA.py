from PIL import Image
import os

indir = 'data/celeba/img_align_celeba_raw'
outdir = 'data/celeba/img_align_celeba'
for f in os.listdir(indir):
    fpath = os.path.join(indir,f)
    img = Image.open(fpath).resize((64,64))
    img.save(os.path.join(outdir,f),'JPEG')
