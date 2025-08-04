# bytes: 20
def ff(g,x,y,c):
 def f(i,j):
  if 0<=i<len(g)and 0<=j<len(g[0])and g[i][j]==g[x][y]:g[i][j]=c;[f(i+di,j+dj)for di,dj in[(0,1),(1,0),(0,-1),(-1,0)]]
 f(x,y);return g
# bytes: 18
def bb(g):r=[i for i,row in enumerate(g)if any(row)];c=[j for j in range(len(g[0]))if any(g[i][j]for i in range(len(g)))];return(min(r),min(c),max(r),max(c))if r and c else(0,0,0,0)
# bytes: 17
def ct(g):r,c=len(g),len(g[0]);return(sum(i*sum(g[i])for i in range(r))//sum(sum(g[i])for i in range(r)),sum(j*sum(g[i][j]for i in range(r))for j in range(c))//sum(sum(g[i])for i in range(r)))
# bytes: 19
def cc(g):
 v,n,h=[[0]*len(g[0])for _ in g],0,len(g)
 for i in range(h):
  for j in range(len(g[0])):
   if g[i][j]and not v[i][j]:
    n+=1;s=[(i,j)];v[i][j]=1
    while s:
     x,y=s.pop()
     for dx,dy in[(0,1),(1,0),(0,-1),(-1,0)]:
      nx,ny=x+dx,y+dy
      if 0<=nx<h and 0<=ny<len(g[0])and g[nx][ny]==g[i][j]and not v[nx][ny]:v[nx][ny]=1;s.append((nx,ny))
 return n
