import os

# with open("ottawa_training.txt", 'w') as f:
#     for i in range(1, 21):
#         im = "ottawa/images/{}.tif".format(i) 
#         gt = "ottawa/gt/{}.png".format(i) 
#         f.write("{}\t{}\n".format(im, gt))
# 
# fnames = os.listdir("test_images")
# with open("test.txt", 'w') as out:
#     for f in fnames:
#         f = "test_images/"+f
#         out.write("{}\t{}\n".format(f, f))

# dirname = "zhang-suen-thinning/edges/"
# fnames = sorted(os.listdir(dirname))
# out_fname = "edges.txt"
# dirname = "zhang-suen-thinning/midlines/"
# fnames = sorted(os.listdir(dirname))
# out_fname = "midlines.txt"
dirname = "test_images/"
out_fname = "test.txt"
fnames = sorted(os.listdir(dirname))
with open(out_fname, 'w') as out:
     for f in fnames:
         f = dirname + f
         out.write("{}\n".format(f))
    

