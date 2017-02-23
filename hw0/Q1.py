import numpy
import sys

aa = numpy.loadtxt(sys.argv[1],delimiter=",")
bb = numpy.loadtxt(sys.argv[2],delimiter=",")


numpy.savetxt("ans_one.txt",numpy.sort(numpy.dot(aa,bb)),delimiter="\n",fmt="%.0f")
