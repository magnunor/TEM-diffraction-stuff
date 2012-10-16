import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
import scipy as sp
import glob

def read_text_image_file(file_name):
  data = np.genfromtxt(file_name)
  return(data)

def create_image(data_array, file_name):
  #figure = pl.figure(figsize=(5,5))
  plt.axes([0,0,1,1]) # Make the plot occupy the whole canvas
  plt.axis('off')
  plt.imshow(data_array,origin='lower')
  #pl.show()
  plt.savefig(file_name, facecolor='black', edgecolor='black', dpi=100, format='png')
  plt.close()

# xCenter,yCenter = 0,0 defined to middle of fourier spectrum
def circle_mask(x, y, data, size):
  dataSize = data.shape

  #x and y is switched because they are defined differently 
  #in matplotlib and the system here
  xCenter = y - dataSize[0]/2
  yCenter = x - dataSize[1]/2

  xrealpos1 = xCenter + dataSize[0]/2
  yrealpos1 = yCenter + dataSize[1]/2
  
  xrealpos2 = -1*xCenter + dataSize[0]/2
  yrealpos2 = -1*yCenter + dataSize[1]/2
  
  y1,x1 = np.ogrid[-xrealpos1:dataSize[0]-xrealpos1, -yrealpos1:dataSize[1]-yrealpos1]
  y2,x2 = np.ogrid[-xrealpos2:dataSize[0]-xrealpos2, -yrealpos2:dataSize[1]-yrealpos2]
  mask1 = x1*x1 + y1*y1 <= size*size
  mask2 = x2*x2 + y2*y2 <= size*size

  maskedData = data*mask1 + data*mask2

  return(maskedData)

# x,y = 0,0 defined to middle of fourier spectrum
def gaussian_mask(x, y, data, size):
  dataSize = data.shape
  maskarray = np.array(dataSize)

  gaussian = gauss_kern(size)

  maskSize = gaussian.shape

  xpadmax = dataSize[0]/2 - maskSize[0]/2 + x - 1
  ypadmax = dataSize[1]/2 - maskSize[1]/2 + y - 1
  xpadmin = dataSize[0]/2 - maskSize[0]/2 - x
  ypadmin = dataSize[1]/2 - maskSize[1]/2 - y

  gaussian = np.append(np.zeros(np.array([xpadmin, maskSize[1]])), gaussian, axis=0)
  gaussian = np.append(gaussian, np.zeros(np.array([xpadmax, maskSize[1]])), axis=0)

  gaussian = np.append(np.zeros(np.array([dataSize[0], ypadmin])), gaussian, axis=1)
  gaussian = np.append(gaussian, np.zeros(np.array([dataSize[0], ypadmax])), axis=1)

  create_image(gaussian, "gaussgri.png")
  
  maskedData = data*gaussian

  return(maskedData)
   
#Returns a normalized 2D gauss kernel array for convolutions
def gauss_kern(size, sizey=None):
  size = int(size)
  if not sizey:
      sizey = size
  else:
      sizey = int(sizey)
  x, y = sp.mgrid[-size:size+1, -sizey:sizey+1]
  g = sp.exp(-(x**2/float(size)+y**2/float(sizey)))
  return g / g.sum() 

#Used to store the position of the mouse clicks in the matplotlib
#plot window
class storeReflections:
  def __init__(self, plot):
    self.plot = plot
    self.xs = []
    self.ys = []
    self.cid = plot.figure.canvas.mpl_connect('button_press_event', self)

  def __call__(self, event):
    if event.inaxes!=self.plot.axes: return
    self.xs.append(event.xdata)
    self.ys.append(event.ydata)

file_list = glob.glob("*.dat")

for file_name in file_list:
  data = read_text_image_file(file_name)
  data_fft = np.fft.fft2(data)
  data_fft_shifted = np.fft.fftshift(data_fft)
  data_ps = np.log(np.abs(np.fft.fftshift(data_fft))**2)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('click to select reflections')
  powerSpectrumPlot = ax.imshow(data_ps)
  reflectionPositions = storeReflections(powerSpectrumPlot)

  plt.show()
  
  reflectionList = []
  
  for reflectionIndex, xPosReflection in enumerate(reflectionPositions.xs):
    yPosReflection = reflectionPositions.ys[reflectionIndex]
    reflectionList.append([xPosReflection, yPosReflection])

  #png_file_name = file_name.replace(".dat",".png")
  #png_fft_file_name = file_name.replace(".dat","_fft.png")
  #png_ps_file_name = file_name.replace(".dat","_ps.png")

 # create_image(np.abs(np.fft.ifft2(data_fft)), "inversFFT.png")

#  create_image(data,png_file_name)

#  create_image(np.abs(data_fft)**3,png_fft_file_name)
#  create_image(data_ps,png_ps_file_name)
#  create_image(data,png_file_name)

#  maskedData = gaussian_mask(0, 0, data_fft, 100)
  braggFilteredList = []
  for reflectionIndex, reflection in enumerate(reflectionList):
    circleMaskedData = circle_mask(reflection[0], reflection[1], data_fft_shifted, 70)
    braggFilteredList.append(np.fft.ifft2(circleMaskedData))
    create_image(np.log(np.abs(circleMaskedData)**2), "circleMaskedData" + str(reflectionIndex) + ".png")

#  gaussianMaskedData = gaussian_mask(0, 0, data_fft_shifted, 250)
#  np.savetxt("maskedFFT.txt", np.abs(maskedData))

  for braggIndex, braggFiltered in enumerate(braggFilteredList):
    create_image(np.abs(braggFiltered), "braggfilter" + str(braggIndex) + ".png")

  for braggIndex, braggFiltered in enumerate(braggFilteredList):
    create_image(np.angle(braggFiltered), "phase" + str(braggIndex) + ".png")

  for braggIndex, braggFiltered in enumerate(braggFilteredList):
    create_image(2*np.real(braggFiltered), "real2" + str(braggIndex) + ".png")

  for braggIndex, braggFiltered in enumerate(braggFilteredList):
    phaseGradient1, phaseGradient2 = np.unwrap(np.gradient(np.angle(braggFiltered)))

#    create_image(phaseGradient1, "phaseGradient1_" + str(braggIndex) + ".png")
#    create_image(phaseGradient2, "phaseGradient2_" + str(braggIndex) + ".png")

#  create_image(np.log(np.abs(gaussianMaskedData)**2), "gaussianMaskedData.png")
