import os
import sys
import open3d
sys.path.insert(0, "../")
from interpolation.interpolation_pcd import interpolation_plant_level
import argparse

os.chdir('./p2p_matching_in_organ')

parser = argparse.ArgumentParser()
parser.add_argument("--type",
                    help="specify the type of plant to run the registration from tomato, maize and arabidopsis, " +
                         "default tomato")
args = parser.parse_args()
if args.type:
    assert args.type in ["tomato", "maize", "arabidopsis"]
    if args.type == "arabidopsis":
        dataset = "lyon2"
    else:
        dataset = args.type
else:
    dataset = "tomato"


if __name__ == "__main__":
    assert dataset in ["tomato", "maize", "lyon2"]
    if dataset == "lyon2":
        days = [ "03-22_PM", "03-23_PM"]
    if dataset == "tomato":
        days = ["03-05_AM", "03-06_AM", "03-07_AM", "03-08_AM", "03-09_AM", "03-10_AM", "03-11_AM"]
    if dataset == "maize":
        days = ["03-13_AM", "03-14_AM", "03-15_AM", "03-16_AM", "03-17_AM", "03-18_AM", "03-20_AM"]
    run_registration = True



    X = []
    i = 1
    for i in range(len(days) - 1):
        day1 = days[i]
        day2 = days[i + 1]

        for alpha in [0.1 * j for j in range(1, 11)]:
            pcd_inter = interpolation_plant_level(day1, day2, dataset, alpha)
            open3d.visualization.draw_geometries([pcd_inter])
            i += 1

