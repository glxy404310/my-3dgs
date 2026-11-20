"""Read-only source validation for the preselected independent Train scene."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from PIL import Image
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--repo',required=True,type=Path)
p.add_argument('--source',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
sys.path.insert(0,str(a.repo))
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary,read_points3D_binary
assert a.source.name == 'train'
assert not a.output.exists()
sparse=a.source/'sparse/0'
ext=read_extrinsics_binary(str(sparse/'images.bin'))
intr=read_intrinsics_binary(str(sparse/'cameras.bin'))
xyz,rgb,error=read_points3D_binary(str(sparse/'points3D.bin'))
assert len(xyz)>0 and np.isfinite(xyz).all()
names=[]
sizes={}
dimensions=set()
archive=zipfile.ZipFile(a.source.parents[2]/'tandt_db.zip')
for image in ext.values():
    camera=intr[image.camera_id]
    assert camera.model in ('PINHOLE','SIMPLE_PINHOLE')
    name=Path(image.name).name
    path=a.source/'images'/name
    assert path.is_file() and path.stat().st_size>0
    with Image.open(path) as pixel:
        width,height=pixel.size
        exact=pixel.size==(camera.width,camera.height)
        half=abs(2*width-camera.width)<=1 and abs(2*height-camera.height)<=1
        assert exact or half, 'Only native size or verified package half-size allowed'
        dimensions.add((camera.width,camera.height,width,height))
        pixel.verify()
    archived=archive.getinfo(f'tandt/train/images/{name}')
    data=path.read_bytes()
    assert len(data)==archived.file_size and zlib.crc32(data)==archived.CRC
    names.append(name)
    sizes[name]=(width,height)
assert len(names)==len(set(names)) and len(names)>=64
names.sort()
test=names[::8]
train=[name for i,name in enumerate(names) if i%8]
assert len(test)>=8 and not set(test)&set(train)
digests={name:hashlib.sha256((sparse/name).read_bytes()).hexdigest() for name in ('images.bin','cameras.bin','points3D.bin')}
for name in digests:
    data=(sparse/name).read_bytes()
    assert zlib.crc32(data)==archive.getinfo(f'tandt/train/sparse/0/{name}').CRC
report=dict(status='pass',schema_version=2,scene='train',source=str(a.source.resolve()),registered_images=len(names),
    train_images=len(train),test_images=len(test),sfm_points=len(xyz),intrinsic_models=sorted({x.model for x in intr.values()}),
    image_sizes=sorted({tuple(x) for x in sizes.values()}),test_names=test,
    colmap_and_actual_dimensions=sorted(dimensions),all_registered_files_match_source_archive_crc=True,
    dimension_scope='Archive images are pre-scaled; native loader keeps COLMAP-derived FOV and uses actual image size. No focal/FOV alteration performed.',
    repo_commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
    colmap_sha256=digests,scope='Input metadata and all registered image files verified. No render scores, method selection, or ground-truth fabrication.')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
