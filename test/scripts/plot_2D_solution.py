import os
import sys

if len(sys.argv) > 1:
    mode = sys.argv[1]
    
    if mode == "debug" or mode == "release":
        sys.path.insert( 1, os.path.join(os.path.dirname(__file__), "..", "classes") )
        
        from Solution import Solution
        
        Solution(mode).plot_soln()
    else:
        sys.exit("Please specify either \"debug\" or \"release\" in the command line.0")