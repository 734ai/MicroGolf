# bytes: 19
"""
Author: Muzan Sano
NeurIPS 2025 ARC-Golf Challenge
License: Apache 2.0
"""
def r90(g):return[list(r)for r in zip(*g[::-1])]
# bytes: 18  
def fh(g):return g[::-1]
# bytes: 20
def fv(g):return[r[::-1]for r in g]
# bytes: 17
def tr(g):return[list(r)for r in zip(*g)]
# bytes: 15
def sh(g,x,y):return g[y:]+g[:y]if x==0 else[r[x:]+r[:x]for r in g]
