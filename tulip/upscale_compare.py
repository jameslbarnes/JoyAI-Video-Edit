"""Create a same-input moving comparison from the completed trial artifacts."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from upscale_trial import ROOT,encode

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--pixel',type=Path,required=True)
parser.add_argument('--latent',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists():raise ValueError('Use a NEW output directory')
args.out.mkdir(parents=True)
readers=[cv2.VideoCapture(str(p)) for p in (args.input,args.pixel,args.latent)]
frames=[];sheet=Image.new('RGB',(1296,5*268))
try:
    for i in range(96):
        current=[]
        for reader in readers:
            ok,frame=reader.read()
            if not ok:break
            current.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if len(current)!=3:break
        row=Image.new('RGB',(1944,392))
        draw=ImageDraw.Draw(row)
        for j,(image,label) in enumerate(zip(current,('Input (display resize)','Pixel SR, denoise .5','Flux latent round-trip SR'))):
            row.paste(Image.fromarray(cv2.resize(image,(648,360),interpolation=cv2.INTER_AREA)),(j*648,32))
            draw.text((j*648+12,10),label,(255,255,255))
        frames.append(np.asarray(row))
        if i in (0,24,48,72,95):
            sheet.paste(row.resize((1296,261)),(0,(0,24,48,72,95).index(i)*268))
        if i==48:
            full=Image.new('RGB',(2592,752));draw=ImageDraw.Draw(full)
            for j,(image,label) in enumerate(zip(current[1:],('Pixel SR .5','Flux latent round-trip'))):
                full.paste(Image.fromarray(image),(j*1296,32));draw.text((j*1296+12,10),label,(255,255,255))
            full.save(args.out/'pixel-vs-latent.png')
finally:
    for reader in readers:reader.release()
if len(frames)<2:raise RuntimeError('Missing comparison videos')
sheet.save(args.out/'motion-contact-sheet.png')
encode(args.out/'comparison.mp4',frames,16,str(ROOT.parent/'easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe'))
print(args.out/'comparison.mp4')
