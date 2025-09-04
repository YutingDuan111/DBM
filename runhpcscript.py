from DBM_HPC import DielectricBreakdown
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    if(len(args) != 5):
        print("Usage: python runhpcscript.py <eta> <triangle_theta> <triangle_height> <max steps> <pickle path>")
        sys.exit()
    
    eta = float(args[0])
    triangle_theta = int(float(args[1]))
    triangle_height = int(float(args[2]))
    max_steps = int(float(args[3]))
    outpath = args[4]

    if(eta < 0.1 or eta > 2 or triangle_theta < 0.1 or triangle_theta > 89):
        print("Allowable ranges for eta: [0.1, 2], triangle_theta: [0.1, 89]")
        print("Please adjust your arguments")
        sys.exit()

    db = DielectricBreakdown(N=4*triangle_height, eta=eta, triangle_theta=triangle_theta, triangle_height=triangle_height)
    db.simulate(max_steps=max_steps, save_incremental=True, save_lite = True, pickle_path=outpath)

