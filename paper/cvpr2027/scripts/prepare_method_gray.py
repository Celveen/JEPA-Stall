#!/usr/bin/env python3
"""Preserve the vector diagram; replace green and export native editable draw.io cells."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import FloatObject
import pypdfium2 as pdfium
from scipy.ndimage import label, find_objects

BASE = 'adbd03b40b8361b41ae64067c76ce88407d77d18'
PALETTE = {'#4f6657':'#293640', '#eef2ee':'#f6f5f2', '#dae4db':'#d7d5cf'}
INK, GRAY, PALE, BORDER = '#293640', '#f6f5f2', '#eef2f4', '#c8ced0'
SCALE = 3


def sha(data):
    return hashlib.sha256(data).hexdigest()


def points(path):
    numbers = [float(x) for x in re.findall(r'[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?', path)]
    return list(zip(numbers[::2],numbers[1::2]))


def css(style):
    return dict(item.strip().split(':',1) for item in style.split(';') if ':' in item)


def neutral_pdf(original,target):
    rgb = lambda color: tuple(bytes.fromhex(color[1:]))
    mapping = {rgb(old):rgb(new) for old,new in PALETTE.items()}
    writer = PdfWriter(clone_from=BytesIO(original))
    page = writer.pages[0]
    stream = page.get_contents()
    operations = stream.operations
    changes = 0
    for index,(operands,operation) in enumerate(operations):
        if operation in (b'rg',b'RG'):
            old = tuple(round(float(x)*255) for x in operands)
            if old in mapping:
                operations[index] = ([FloatObject(x/255) for x in mapping[old]],operation)
                changes += 1
    stream.operations = operations
    page.replace_contents(stream)
    page.compress_content_streams(level=9)
    temporary = target.with_suffix('.temporary.pdf')
    writer.write(temporary)
    temporary.replace(target)
    return changes


def make_drawio(svg,target):
    source = ET.fromstring(svg)
    document = ET.Element('mxfile',host='app.diagrams.net',version='26.0.9',type='device')
    diagram = ET.SubElement(document,'diagram',id='ecp-method',name='Figure 3 — editable')
    graph = ET.SubElement(diagram,'mxGraphModel',dx='1404',dy='702',grid='1',gridSize='6',
                          guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',
                          pageScale='1',pageWidth='1404',pageHeight='702',math='0',shadow='0')
    root = ET.SubElement(graph,'root')
    ET.SubElement(root,'mxCell',id='0')
    ET.SubElement(root,'mxCell',id='1',parent='0')
    count = {'shapes':0,'edges':0,'text':0,'embedded_simulator_images':0}

    def vertex(identifier,x,y,w,h,style,text=''):
        cell = ET.SubElement(root,'mxCell',id=identifier,value=text,style=style,vertex='1',parent='1')
        ET.SubElement(cell,'mxGeometry',x=f'{x*SCALE:g}',y=f'{y*SCALE:g}',
                      width=f'{w*SCALE:g}',height=f'{h*SCALE:g}',**{'as':'geometry'})

    def shape(identifier,x,y,w,h,fill,stroke=INK,lw=.8,ellipse=False,rounded=False):
        vertex(identifier,x,y,w,h,
               f'shape={"ellipse" if ellipse else "rectangle"};rounded={int(rounded)};arcSize=12;'
               f'fillColor={fill};strokeColor={stroke};strokeWidth={lw*SCALE:g};html=1;')
        count['shapes'] += 1

    def edge(identifier,path,color=INK,lw=1.5,arrow=True):
        cell = ET.SubElement(root,'mxCell',id=identifier,value='',edge='1',parent='1',
                             style=f'edgeStyle=none;rounded=0;html=1;strokeColor={color};'
                                   f'strokeWidth={lw*SCALE:g};endArrow={"block" if arrow else "none"};endFill=1;endSize=8;')
        geometry = ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
        for name,point in [('sourcePoint',path[0]),('targetPoint',path[-1])]:
            ET.SubElement(geometry,'mxPoint',x=f'{point[0]*SCALE:g}',y=f'{point[1]*SCALE:g}',**{'as':name})
        if len(path)>2:
            array = ET.SubElement(geometry,'Array',**{'as':'points'})
            for x,y in path[1:-1]:
                ET.SubElement(array,'mxPoint',x=f'{x*SCALE:g}',y=f'{y*SCALE:g}')
        count['edges'] += 1

    boxes = [('offline',3,3,462,41,GRAY,'#d7d5cf',.6),
             ('learning',169,15,119,24,PALE,INK,.8),
             ('world_model',319,15,137,24,'#ffffff',INK,.8),
             ('observations',4,66,84,73,'#ffffff',BORDER,.8),
             ('planner',170,66,174,73,PALE,INK,.8),
             ('execution',366,66,98,73,'#ffffff',BORDER,.8),
             ('feedback',353,162,111,62,GRAY,INK,.8),
             ('detector',224,162,109,62,GRAY,INK,.8),
             ('correction',4,162,202,62,GRAY,INK,.8),
             ('encoder',107,86,43,35,PALE,INK,1.15)]
    for identifier,x,y,w,h,fill,stroke,lw in boxes:
        shape(identifier,x,y,w,h,fill,stroke,lw,rounded=True)
    shape('training_agent',15,28,6,6,'#748b9b',lw=.7,ellipse=True)
    shape('training_object',34,22,7,17,'#a2b2ba',lw=.7)
    shape('memory_neighborhood',21,197,18,18,'#d7d5cf',ellipse=True)
    for i,(x,y) in enumerate([(26,205),(29,209),(33,203)]):
        shape(f'memory_point_{i}',x-1.6,y-1.6,3.2,3.2,INK,lw=0,ellipse=True)
    for i,(x,y,w,h,color) in enumerate([(132,206,6,10,INK),(142,213,6,3,'#88664e'),
                                       (174,206,6,10,INK),(184,198,6,18,'#88664e')]):
        shape(f'component_bar_{i}',x,y,w,h,color,'none',0)

    for group in source.iter():
        identifier = group.attrib.get('id','')
        if not (identifier.startswith('patch_') and int(identifier.split('_')[1])>=25
                or identifier.startswith('line2d_')):
            continue
        paths = [element for element in group if element.tag.endswith('path')]
        if not paths:
            continue
        path = points(paths[0].attrib.get('d',''))
        if len(path)<2:
            continue
        style = css(paths[0].attrib.get('style',''))
        arrow = identifier.startswith('patch_')
        if arrow and len(paths)>1:
            head = points(paths[-1].attrib.get('d',''))
            direction = np.asarray(path[-1])-path[-2]
            if head:
                path[-1] = max(head,key=lambda p:np.dot(np.asarray(p)-path[-2],direction))
        edge('vector_'+identifier,path,style.get('stroke',INK).strip(),float(style.get('stroke-width',1)),arrow)
    shape('latent_start',223.3,108.3,3.4,3.4,INK,lw=0,ellipse=True)
    vertex('latent_goal',312.5,83.5,7,7,f'shape=mxgraph.basic.star;fillColor={INK};strokeColor={INK};')
    count['shapes'] += 1

    image_hashes = []
    for element in source.iter():
        if not element.tag.endswith('image'):
            continue
        uri = element.attrib['{http://www.w3.org/1999/xlink}href']
        payload = base64.b64decode(uri.split(',',1)[1])
        image_hashes.append(sha(payload))
        image = Image.open(BytesIO(payload)).convert('RGBA')
        regions,_ = label(np.asarray(image.getchannel('A'))>0)
        for region in find_objects(regions):
            if region is None:
                continue
            y0,y1 = region[0].start,region[0].stop
            x0,x1 = region[1].start,region[1].stop
            if (y1-y0)*(x1-x0)<=1000:
                continue
            width,height = float(element.attrib['width']),float(element.attrib['height'])
            x = float(element.attrib['x'])+x0/image.width*width
            y = -float(element.attrib['y'])+(image.height-y1)/image.height*height
            cropped = image.crop((x0,y0,x1,y1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            png = BytesIO();cropped.save(png,format='PNG')
            encoded = base64.b64encode(png.getvalue()).decode()
            vertex(f'native_frame_{count["embedded_simulator_images"]}',x,y,
                   (x1-x0)/image.width*width,(y1-y0)/image.height*height,
                   f'shape=image;imageAspect=0;aspect=fixed;html=1;image=data:image/png,{encoded};')
            count['embedded_simulator_images'] += 1
    for i,(x,y,w,h) in enumerate([(10,88,31,31),(50,88,31,31),(377,89,29,29),
                                 (423,89,29,29),(364,184,25,25),(426,184,25,25)]):
        shape(f'frame_border_{i}',x,y,w,h,'none',BORDER,.6)

    def text(identifier,content,x,baseline,font=8.5,centered=True,bold=False,width=None):
        width = width or max(10,len(re.sub('<[^>]+>','',content))*font*.52)
        left = x-width/2 if centered else x
        vertex(identifier,left,baseline-font*.87,width,font*1.1,
               f'text;html=1;whiteSpace=nowrap;overflow=visible;align={"center" if centered else "left"};'
               f'verticalAlign=middle;resizable=1;fontFamily=Times New Roman;fontColor={INK};'
               f'fontSize={font*SCALE:g};fontStyle={int(bold)};spacing=0;',content)
        count['text'] += 1

    for group in source.iter():
        if not group.attrib.get('id','').startswith('text_'):
            continue
        for i,element in enumerate(group):
            if not element.tag.endswith('text') or list(element):
                continue
            style = element.attrib.get('style','')
            size = re.search(r'([0-9.]+)px',style)
            font = float(size.group(1)) if size else 8.5
            x,y = float(element.attrib.get('x',0)),float(element.attrib.get('y',0))
            translate = re.search(r'translate\(([-0-9.]+)[ ,]+([-0-9.]+)\)',element.attrib.get('transform',''))
            if translate:
                x+=float(translate.group(1));y+=float(translate.group(2))
            text(group.attrib['id']+f'_{i}',element.text or '',x,y,font,
                 centered='text-anchor: middle' in style,bold='700 ' in style)
    math_labels = [
        ('model_symbols','Encoder <i>E</i><sub>θ</sub> + predictor <i>P</i><sub>θ</sub>',387.5,34.9,8,112),
        ('state_symbol','<i>o</i><sub>t</sub>',25.5,132,10,18),
        ('goal_symbol','<i>o</i><sub>g</sub>',65.5,132,10,18),
        ('encoder_symbol','<i>E</i><sub>θ</sub>',128,106.3,14,28),
        ('latent_symbols','<i>z</i><sub>t</sub>, <i>z</i><sub>g</sub>',158,86.5,9,30),
        ('predictor_symbol','<i>P</i><sub>θ</sub>',190,99.8,13,23),
        ('latent_goal_symbol','<i>z</i><sub>g</sub>',323,101.4,9,18),
        ('objective','<i>J</i><sub>goal</sub>(<i>A</i>; α<sub>t</sub>) + κg<sub>t</sub>ℛ<sub>t</sub><sup>stall</sup>(<i>A</i>)',257,127.8,9.5,151),
        ('action_symbol','<i>a</i><sub>t</sub>',356.5,91.1,10,18),
        ('executed_action_symbol','<i>a</i><sub>t</sub>',408.5,189.9,9,18),
        ('transition_before','<i>o</i><sub>t</sub>',377,217.9,9,18),
        ('transition_after','<i>o</i><sub>t+1</sub>',438,217.9,9,24)]
    for identifier,content,x,y,font,width in math_labels:
        text(identifier,content,x,y,font,width=width)
    ET.indent(document,space='  ')
    target.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(document).write(target,encoding='utf-8',xml_declaration=True)
    assert count['embedded_simulator_images']==6
    assert count['edges']>=15 and count['text']>=30
    assert not any(color in target.read_text() for color in PALETTE)
    return count,image_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    paper=args.paper.resolve();repo=paper.parents[1]
    source_dir=paper/'figures/source';source_dir.mkdir(parents=True,exist_ok=True)
    sources={}
    for ext,relative in [('pdf','figures/fig2_method.pdf'),('svg','figures/svg/fig2_method.svg')]:
        path=source_dir/f'fig2_method_v14.{ext}'
        if not path.exists():
            path.write_bytes(subprocess.check_output(['git','-C',str(repo),'show',f'{BASE}:paper/cvpr2027/{relative}']))
        sources[ext]=path.read_bytes()
    original_manifest=json.loads((paper/'provenance/method_figure.json').read_text())
    pdf,svg,png=(paper/'figures/fig2_method.pdf',paper/'figures/svg/fig2_method.svg',paper/'figures/previews/fig2_method.png')
    changes=neutral_pdf(sources['pdf'],pdf)
    text_svg=sources['svg'].decode()
    for old,new in PALETTE.items():text_svg=text_svg.replace(old,new)
    svg.write_text(text_svg)
    document=pdfium.PdfDocument(pdf);document[0].render(scale=4).to_pil().save(png);document.close()
    drawio=repo/'delivery/Figure3.drawio'
    counts,images=make_drawio(text_svg,drawio)
    original_images=[sha(base64.b64decode(e.attrib['{http://www.w3.org/1999/xlink}href'].split(',',1)[1]))
                     for e in ET.fromstring(sources['svg']).iter() if e.tag.endswith('image')]
    assert images==original_images
    manifest={'canvas_pt':[468,234],'base_commit':BASE,'base_pdf_sha256':sha(sources['pdf']),
              'base_svg_sha256':sha(sources['svg']),'palette':PALETTE,'pdf_color_operators_changed':changes,
              'layout_and_text_preserved':True,'native_simulator_image_payloads_unchanged':True,
              'native_frame_hashes':original_manifest.get('native_frame_hashes',{}),
              'drawio':{'path':str(drawio.relative_to(repo)),'native_editable_cells':counts,
                        'embedded_original_simulator_views':6,'external_images':False},
              'outputs':{str(p.relative_to(repo)):sha(p.read_bytes()) for p in [pdf,svg,png,drawio]},
              'generator_sha256':sha(Path(__file__).read_bytes())}
    (paper/'provenance/method_figure.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'pdf_color_changes':changes,'drawio_editable_cells':counts,'native_images_unchanged':True}))


if __name__=='__main__':
    main()
