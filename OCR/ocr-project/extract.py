import xml.etree.ElementTree as ET
import glob, os, re

os.makedirs('../plaintext', exist_ok=True)
for f in sorted(glob.glob('OCR-D-OCR/*.xml')):
    root = ET.parse(f).getroot()
    uri = re.match(r'\{(.*)\}', root.tag).group(1)
    def q(t): return '{%s}%s' % (uri, t)
    lines = []
    for tl in root.iter(q('TextLine')):
        ws = [u.text for w in tl.findall(q('Word'))
              for u in w.findall('%s/%s' % (q('TextEquiv'), q('Unicode')))
              if u.text and u.text.strip()]
        if ws:
            lines.append(' '.join(ws))
        else:
            u = tl.find('%s/%s' % (q('TextEquiv'), q('Unicode')))
            if u is not None and u.text and u.text.strip():
                lines.append(u.text.strip())
    out = '../plaintext/%s.txt' % os.path.basename(f)[:-4]
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('%s: %d lines' % (os.path.basename(out), len(lines)))
