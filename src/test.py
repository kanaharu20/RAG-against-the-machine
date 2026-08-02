import re

for m in re.finditer(r'\d+', 'a1 b22 c333'):
    print(m.group(), m.start(), m.end())
