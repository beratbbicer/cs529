import math
x = [1,1,  0.5,0,0.7,1,1,0,  0,1,0,2,8,5,0,3,0,3,0,   1,0]
y = [4,0.3,0,  0,9,  0,0,0.1,6,7,0,3,0,0,0,1,0,0,0.75,0,0]

xmean = sum(x) / float(len(x))
ymean = sum(y) / float(len(y))

print(f'xmean: {xmean}')
print(f'ymean: {ymean}')

f = lambda a,m: (a - m)**2
divider1 = math.sqrt(sum([f(i,xmean) for i in x]))
divider2 = math.sqrt(sum([f(i,ymean) for i in y]))
print(f'divider1: {divider1}')
print(f'divider2: {divider2}')

top = sum([(i-xmean)*(j-ymean) for i,j in zip(x,y)])
print(f'Top: {top}')

pearson = top / (divider1 *divider2)
print(f'Pearson: {pearson}')