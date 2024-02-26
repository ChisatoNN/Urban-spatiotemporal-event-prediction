import shapefile
import matplotlib.pyplot as plt

sf = shapefile.Reader('.roda network.shp')
shapes = sf.shapes()
codes = []
pts = shapes[0].points

x,y = zip(*pts)
fig = plt.figure(figsize=[12,18])
ax = fig.add_subplot(111)
ax.plot(x, y, '-', lw=1, color='k')
plt.show()
