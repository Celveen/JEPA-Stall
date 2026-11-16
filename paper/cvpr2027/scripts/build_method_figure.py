#!/usr/bin/env python3
"""Refine the supplied v2 vector figure and keep its draw.io source editable."""
import argparse
import base64
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import pymupdf as fitz
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab
import pypdfium2 as pdfium

INK='#293640'; PALE='#f6f5f2'; BLUE='#91a7b5'; LIGHT='#e3e7e8'; ACCENT='#9b735a'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,default=Path(__file__).resolve().parents[1])
    paper=parser.parse_args().paper.resolve(); repo=paper.parents[1]
    source=paper/'figures/source'; src_pdf=source/'fig2_method_v2_input.pdf'; src_xml=source/'Figure3_v2_input.drawio'
    document=fitz.open(src_pdf); page=document[0]
    # Clear the two panel interiors while retaining their borders and all other artwork.
    regions=[(6,18.2,141,42),(5,163,205,223),(225,163,332,223),(364,64,467,161)]
    for box in regions:page.add_redact_annot(fitz.Rect(box),fill=((1,1,1) if box[0]==364 else tuple(int(PALE[i:i+2],16)/255 for i in (1,3,5))),cross_out=False)
    page.apply_redactions(images=0,graphics=1,text=0)
    tree=ET.parse(src_xml); root=tree.getroot().find('.//root')
    def inside(x,y,box):return box[0]<=x<=box[2] and box[1]<=y<=box[3]
    removed=[]
    for cell in list(root):
        g=cell.find('mxGeometry')
        if g is None:continue
        pts=[]
        if cell.get('vertex')=='1':
            x,y,ww,hh=[float(g.get(k,'0'))/3 for k in ('x','y','width','height')];pts=[(x,y),(x+ww,y+hh)]
        elif cell.get('edge')=='1':pts=[(float(p.get('x','0'))/3,float(p.get('y','0'))/3) for p in g.iter('mxPoint')]
        if pts and any(all(inside(x,y,box) for x,y in pts) for box in regions):root.remove(cell);removed.append(cell.get('id'))
    fonts=Path('/System/Library/Fonts/Supplemental')
    for name,filename,backup in [('Diagram','Times New Roman.ttf','Vera.ttf'),('DiagramBold','Times New Roman Bold.ttf','VeraBd.ttf'),('DiagramItalic','Times New Roman Italic.ttf','VeraIt.ttf')]:
        p=fonts/filename
        if not p.exists():p=Path(reportlab.__file__).parent/'fonts'/backup
        pdfmetrics.registerFont(TTFont(name,str(p)))
    overlay=BytesIO(); c=canvas.Canvas(overlay,pagesize=(468,234),invariant=1);serial=0
    def vertex(x,y,w,h,style,value=''):
        nonlocal serial
        serial+=1;v=ET.SubElement(root,'mxCell',id=f'planning_v2_{serial}',value=value,style=style,vertex='1',parent='1')
        ET.SubElement(v,'mxGeometry',x=str(x*3),y=str(y*3),width=str(w*3),height=str(h*3),attrib={'as':'geometry'})
    def text(x,y,s,size=8,bold=False,italic=False,align='center',color=INK):
        font='DiagramBold' if bold else 'DiagramItalic' if italic else 'Diagram';c.setFont(font,size);c.setFillColor(color)
        if align=='center':c.drawCentredString(x,234-y,s)
        else:c.drawString(x,234-y,s)
        width=pdfmetrics.stringWidth(s,font,size)+2;left=x-width/2 if align=='center' else x
        vertex(left,y-size*.86,width,size*1.12,f'text;html=1;whiteSpace=nowrap;overflow=visible;align={align};verticalAlign=middle;fontFamily=Times New Roman;fontSize={size*3};fontStyle={1 if bold else 2 if italic else 0};fontColor={color};spacing=0;',s)
    def rect(x,y,w,h,fill,stroke=INK,lw=.45):
        c.setFillColor(fill);c.setStrokeColor(stroke);c.setLineWidth(lw);c.rect(x,234-y-h,w,h,fill=1,stroke=int(lw>0))
        vertex(x,y,w,h,f'shape=rectangle;fillColor={fill};strokeColor={stroke if lw else "none"};strokeWidth={lw*3};')
    def ellipse(x,y,w,h,fill,stroke=INK,lw=.6):
        c.setFillColor(fill);c.setStrokeColor(stroke);c.setLineWidth(lw);c.ellipse(x,234-y-h,x+w,234-y,fill=1,stroke=int(lw>0))
        vertex(x,y,w,h,f'shape=ellipse;fillColor={fill};strokeColor={stroke if lw else "none"};strokeWidth={lw*3};')
    def line(pts,color=INK,lw=.8,arrow=False,dashed=False):
        nonlocal serial
        c.setStrokeColor(color);c.setLineWidth(lw);c.setDash(2,2) if dashed else c.setDash()
        p=c.beginPath();p.moveTo(pts[0][0],234-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,234-y)
        c.drawPath(p);c.setDash()
        if arrow:
            x,y=pts[-1];a=math.atan2(y-pts[-2][1],x-pts[-2][0]);length=3
            points=[(x,y),(x-length*math.cos(a-.45),y-length*math.sin(a-.45)),(x-length*math.cos(a+.45),y-length*math.sin(a+.45))]
            q=c.beginPath();q.moveTo(points[0][0],234-points[0][1])
            for xx,yy in points[1:]:q.lineTo(xx,234-yy)
            q.close();c.setFillColor(color);c.drawPath(q,fill=1,stroke=0)
        serial+=1;v=ET.SubElement(root,'mxCell',id=f'planning_v2_{serial}',value='',style=f'edgeStyle=none;rounded=0;strokeColor={color};strokeWidth={lw*3};endArrow={"block" if arrow else "none"};endSize=6;dashed={int(dashed)};',edge='1',parent='1')
        g=ET.SubElement(v,'mxGeometry',relative='1',attrib={'as':'geometry'})
        for k,(x,y) in [('sourcePoint',pts[0]),('targetPoint',pts[-1])]:ET.SubElement(g,'mxPoint',x=str(x*3),y=str(y*3),attrib={'as':k})
        if len(pts)>2:
            arr=ET.SubElement(g,'Array',attrib={'as':'points'})
            for x,y in pts[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x*3),y=str(y*3))
    # Four task thumbnails, sourced from native execution-state renders / captured RGB.
    c.setFillColor('white');c.setStrokeColor('#c8d0d3');c.setLineWidth(.8)
    c.roundRect(366,234-151,98,85,3,fill=1,stroke=1)
    vertex(366,66,98,85,'rounded=1;arcSize=8;fillColor=#ffffff;strokeColor=#c8d0d3;strokeWidth=2.4;')
    text(415,76,'3  Execute first block',8.8,bold=True)
    for name,label,x,y in [('pushobj','PushObj',374,81),('maze','Maze',426,81),('pusht','PushT shift',374,116),('rope','Rope',426,116)]:
        path=paper/'figures/domains_20261116'/f'diagram_{name}.png'
        c.drawImage(str(path),x,234-y-26,26,26,mask='auto')
        vertex(x,y,26,26,'shape=image;imageAspect=1;aspect=fixed;html=1;image=data:image/png,'+base64.b64encode(path.read_bytes()).decode()+';')
        text(x+13,y+32,label,7.2)
    line([(415,152),(415,159)],INK,1.5,True)
    # Collection-time contact bias plus perturbations; no low-contrast algorithm tag.
    ellipse(11.5,27.5,5,5,BLUE,INK,.6)
    rect(35,22,6,17,'#adbcc3',INK,.7)
    line([(17.5,30),(31.5,24)],BLUE,.8,True)
    line([(17.5,30),(30.5,36)],BLUE,.8,True)
    line([(17.5,30),(32,30)],ACCENT,1.25,True)
    text(49,26.8,'Contact bias + noise',8.4,align='left')
    text(49,37,'Collect transitions',8.4,align='left')
    # Detection: a large local path with almost no start-to-end displacement.
    text(278.5,172.2,'4  Detect stagnation',8.8,bold=True)
    line([(244,198),(250,184),(259,195),(246,203),(240,190),(254,190),(246,198)],BLUE,1.1)
    ellipse(242.7,196.7,2.6,2.6,INK,lw=0);ellipse(245,197,2,2,ACCENT,lw=0)
    line([(244,198),(247,198)],INK,1,True)
    line([(250,197),(307,191)],BLUE,.65,dashed=True)
    star=[]
    for i in range(11):
        angle=-math.pi/2+i*math.pi/5;rad=3.5 if i%2==0 else 1.5;star.append((314+rad*math.cos(angle),191+rad*math.sin(angle)))
    line(star,INK,.9)
    text(314,200,'goal',7.3)
    text(278.5,210,'Small / repeated move',7.8)
    text(278.5,220,'→ Activate correction',7.8)
    # Temporal organization: persistent memory and a three-call weighting interval.
    text(105,172.2,'5  Correct next plans',8.8,bold=True)
    ellipse(16.5,186.5,11,11,'#d7d5cf')
    for x,y in [(19,191),(22,193),(24,189)]:ellipse(x,y,1.8,1.8,INK,lw=0)
    text(35,194.2,'Avoid revisits (S)',7.5,align='left')
    rect(17,208,3.5,5,INK,lw=0);rect(24,202,3.5,11,ACCENT,lw=0)
    text(35,211,'Boost goal weight (P)',7.5,align='left')
    for i,label in enumerate(['t','t+1','t+2','t+3','...']):
        x=115+i*17;text(x+6.5,183,label,7.2,italic=True)
        rect(x,187,13,9,BLUE,BLUE,.3)
        rect(x,204,13,9,ACCENT if i<3 else LIGHT,ACCENT if i<3 else '#cad0d3',.3)
    text(155,221,'Across replans',7.5)
    c.save();ov=fitz.open(stream=overlay.getvalue(),filetype='pdf');page.show_pdf_page(page.rect,ov,0)
    target=paper/'figures/fig2_method.pdf';document.subset_fonts();document.save(target,garbage=4,deflate=True);document.close();ov.close()
    d=fitz.open(target);(paper/'figures/svg/fig2_method.svg').write_text(d[0].get_svg_image(text_as_path=True));d.close()
    d=pdfium.PdfDocument(target);preview=paper/'figures/previews/fig2_method.png';d[0].render(scale=4).to_pil().save(preview);d.close()
    drawio=repo/'delivery/Figure3.drawio';ET.indent(tree,space='  ');tree.write(drawio,encoding='utf-8',xml_declaration=True); (paper/'Figure3.drawio').write_bytes(drawio.read_bytes())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'source_type':'supplied_v2_with_vector_refinements','canvas_pt':[468,234],'sources':{str(p.relative_to(paper)):sha(p) for p in (src_pdf,src_xml,*sorted((paper/'figures/domains_20261116').glob('diagram_*.png')))},'changes':['Step 4: trajectory with Small / repeated move -> Activate correction explanation','Step 5: Avoid revisits and Boost goal weight with persistence timeline','Offline: contact-directed and perturbed action arrows with Contact bias + noise / Collect transitions; no algorithm tag','Step 3: PushObj, Maze, PushT color shift and recorded Rope simulation thumbnails'], 'removed_editable_cells':removed,'new_editable_cells':serial,'outputs':{str(p.relative_to(repo)):sha(p) for p in (target,preview,paper/'figures/svg/fig2_method.svg',drawio)},'generator_sha256':sha(Path(__file__))}
    (paper/'provenance/method_figure.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Refined vector Figure 3 and editable diagram generated.')

if __name__=='__main__':main()
