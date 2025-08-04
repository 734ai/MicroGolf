# bytes: 15
def inc(g):return[[c+1 for c in r]for r in g]
# bytes: 18
def cl(g,mi,ma):return[[max(mi,min(ma,c))for c in r]for r in g]
# bytes: 20
def he(g):h={};[[h.update({c:h.get(c,0)+1})for c in r]for r in g];s=sorted(h.items(),key=lambda x:x[1]);m={s[i][0]:i for i in range(len(s))};return[[m.get(c,c)for c in r]for r in g]
# bytes: 16
def sm(g):return sum(sum(r)for r in g)
# bytes: 17
def avg(g):t=sm(g);return t//(len(g)*len(g[0]))
