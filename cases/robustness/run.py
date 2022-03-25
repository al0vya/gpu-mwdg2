import os
import sys
import subprocess

def EXIT_HELP():
    help_message = (
        "This tool is used to verify the robustness of a solver. Run using:\n" +
        "python run.py <SOLVER>, SOLVER={hw|mw}"
    )
    
    sys.exit(help_message)

if __name__ == "__main__":
    if len(sys.argv) < 2: EXIT_HELP()
    
    dummy, solver = sys.argv
    
    if solver != "hw" and solver != "mw": EXIT_HELP()
    
    with open("tests.txt", 'w') as fp:
        fp.write("19\n20\n21")
        
    test_script = os.path.join("..", "tests", "test.py")
    
    subprocess.run( ["python", test_script, "test", solver, "1e-3", "7", "surf"] )
    subprocess.Popen( ["matlab", "-nosplash", "-nodesktop", "-r", "\"main; exit\""] )