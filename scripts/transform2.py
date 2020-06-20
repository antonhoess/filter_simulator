#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
import sys
import getopt

verbose = False


def main(argv):
    global verbose

    fn_in = None
    fn_out = None
    swap = False
    coord_sys = "ENU"
    show = False

    steps = list()
    ids = list()

    # Read and check parameters
    ###########################
    usage_info = "Usage: " + sys.argv[0] + " [-i INPUT_FILE] [-o OUTPUT_FILE] [-w] [-s] [-v] \n" \
                                           "Converts format 'X Y STEP' to YAML-format V1.0."

    try:
        opts, args = getopt.getopt(argv, "hi:o:wc:sv")

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

        elif opt == "-w":
            swap = True

        elif opt == "-c":
            coord_sys = arg

        elif opt == "-s":
            show = True

        elif opt == "-v":
            verbose = True
        # end if
    # end for

    if fn_in is None:
        fn_in = 0  # STDIN
    # end if

    if fn_out is None:
        fn_out = 1  # STDOUT
    # end if

    # Read coordinates from input file
    ##################################
    with open(fn_in, 'r') as file_in:
        count = 0
        while True:
            # Get next line from file
            line = file_in.readline().strip()
            count += 1

            # if line is empty end of file is reached
            if not line:
                break

            else:
                if verbose:
                    print("Line{}: {}".format(count, line.strip()))

                fields = line.split(" ")

                if len(fields) == 3:  # Format: X, Y, STEP
                    if len(ids) == 0 or fields[2] != ids[-1]:  # A new time step
                        ids.append(fields[2])
                        steps.append(list())
                    # end if

                    x, y = float(fields[0]), float(fields[1])

                    if swap:
                        x, y = y, x
                    steps[-1].append({"x": x, "y": y})
                # end if
            # end if
        # end while
    # end with

    # Write reformatted coordinates to output file
    ##############################################
    if fn_out is not None:
        with open(fn_out, 'w') as file_out:
            file_out.write("meta-information:\n")
            file_out.write("  version: '1.0'\n")
            file_out.write("  number-steps: {}\n".format(len(steps)))
            file_out.write("  coordinate-system: {}\n".format(coord_sys))
            file_out.write("  time-delta: 1.0\n")
            file_out.write("\n")
            file_out.write("detections:\n")

            for step in steps:
                first_det_in_step = True

                for set in step:
                    if first_det_in_step:
                        first_det_in_step = False
                        step_dash = "- "

                    else:
                        step_dash = "  "
                    # end if

                    file_out.write("{}- x: {}{}".format(step_dash, set["x"], os.linesep))
                    file_out.write("{}  y: {}{}".format("  ", set["y"], os.linesep))
                # end for
            # end for
        # end with
    # end if

    # Show coordinates as a scatter plot
    ####################################
    if show:
        show_coords(steps)
    # end if
# end def main


def show_coords(steps):
    global verbose

    if verbose:
        print("# steps: {}".format(len(steps)))
    # end if

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # All detections - each frame's detections in a different color
    for step in steps:
        ax.scatter([det["x"] for det in step], [det["y"] for det in step], s=5, linewidth=.5, edgecolor="green", marker="o", zorder=0)
    # end for

    # Connections between all detections - only makes sense, if they are manually created or created in a very ordered way, otherwise it's just chaos
    for step in steps:
        ax.plot([det["x"] for det in step], [det["y"] for det in step], color="black", linewidth=.5, linestyle="--", zorder=1)
    # end for

    ax.set_aspect("equal")

    plt.show()
# end def


if __name__ == "__main__":
    main(sys.argv[1:])
