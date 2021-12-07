

if len(sys.argv) > 1:
    solver = sys.argv[1]

    if solver != "hw" and solver != "mw":
        print("Please specify either \"hw\" or \"mw\" in the command line.")
    else:
        DischargeErrors(solver).plot_errors(0, "ad-hoc")