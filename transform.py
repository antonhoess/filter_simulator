#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
import sys
import getopt


verbose = False


def main(argv):
  fn_in = None
  fn_out = None
  show = False

  coords_x = []
  coords_y = []

  ## Read and check parameters
  ############################
  usage_info = "Usage: " + sys.argv[0] + " -i INPUT_FILE [-o OUTPUT_FILE] [-s] [-v]"

  try:
    opts, args = getopt.getopt(argv,"hi:o:sv")

  except getopt.GetoptError:
    print(usage_info)
    sys.exit(2)
  # end try

  for opt, arg in opts:
    if opt == '-h':
      print(usage_info)
      sys.exit()

    elif opt == "-i":
      fn_in = arg

    elif opt == "-o":
      fn_out = arg

    elif opt == "-s":
      show = True

    elif opt == "-v":
      verbose = True
    # end if
  # end for

  if fn_in is None:
    print("No input file specified.")
    print(usage_info)
    exit(1)
  # end if

  ## Read coordinates from input file
  ###################################
  with open(fn_in, 'r') as file:
    count = 0
    while True:
      # Get next line from file
      line = file.readline()
      count += 1

      # if line is empty end of file is reached
      if not line:
        break

      else:
        if verbose:
          print("Line{}: {}".format(count, line.strip()))

        fields = line.split(" ")

        if len(fields) == 3 and float(fields[0]) != .0:  # Only valid lines and no 0-GPS-pseudo-positions
          coords_x.append(float(fields[0]))
          coords_y.append(float(fields[1]))
        # end if
      # end if
    # end while
  # end with

  ## Write reformatted coordinates to output file
  ###############################################
  if fn_out is not None:
    with open(fn_out, 'w') as file:
      for i in range(len(coords_x)):
        file.write("{} {}{}".format(coords_x[i], coords_y[i], os.linesep))
      # end for
    # end with
  # end if

  ## Show coordinates as a scatter plot
  #####################################
  if show:
    show_coords(coords_x, coords_y)
  # end if
# end def main


def show_coords(coords_x, coords_y):
  if verbose:
    print(len(coords_x))  # Same as length of coords_y

  if verbose:
    print(coords_x)
    print(coords_y)
  # end if

  step = 5
  plt.scatter(coords_x[0::step], coords_y[0::step], c='red')
  plt.show()
# end def


if __name__ == "__main__":
   main(sys.argv[1:])
