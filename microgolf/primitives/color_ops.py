# bytes: 20
def mc(g,m):return[[m.get(c,c)for c in r]for r in g]
# bytes: 18
def tm(g,t):return[[1 if c>t else 0 for c in r]for r in g]
# bytes: 17
def rc(g,o,n):return[[n if c==o else c for c in r]for r in g]
# bytes: 19
def bc(g,a,b):return[[int(c*a+b)for c in r]for r in g]
# bytes: 16
def md(g):c=max(max(r)for r in g);return[[c-x for x in r]for r in g]
