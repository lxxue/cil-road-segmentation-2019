import matplotlib.pyplot as plt
import imageio

for i in range(1, 101):
    edge = imageio.imread("edges/satImage_%03d.png" % i)
    midline = imageio.imread("midlines/satImage_%03d.png" % i)
    plt.subplot(121),plt.imshow(edge, cmap='gray')
    print(edge.shape)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(midline, cmap='gray')
    print(midline.shape)
    plt.title('Midline Image'), plt.xticks([]), plt.yticks([])
    plt.show()